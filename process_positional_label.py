import numpy as np
import cv2
from tqdm import tqdm
import json
from shapely.geometry import Polygon
import copy
from scipy.special import comb as n_over_k
import torch
import sys


def convert_bezier_ctrl_pts_to_polygon(bez_pts, sample_num_per_side):
    '''
    An example of converting Bezier control points to polygon points for a text instance.
    The generation of Bezier label can be referred to https://github.com/Yuliang-Liu/bezier_curve_text_spotting

    Args:
        bez_pts (np.array): 8 Bezier control points in clockwise order, 4 for each side (top and bottom).
                            The top side is in line with the reading order of this text instance.
                            [x_top_0, y_top_0,.., x_top_3, y_top_3, x_bot_0, y_bot_0,.., x_bot_3, y_bot_3].
        sample_num_per_side (int): Sampled point numbers on each side.

    Returns:
        sampled_polygon (np.array): The polygon points sampled on Bezier curves.
                                    The order is the same as the Bezier control points.
                                    The shape is (2 * sample_num_per_side, 2).
    '''
    Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
    BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]
    assert (len(bez_pts) == 16), 'The numbr of bezier control points must be 8'
    s1_bezier = bez_pts[:8].reshape((4, 2))
    s2_bezier = bez_pts[8:].reshape((4, 2))
    t_plot = np.linspace(0, 1, sample_num_per_side)
    Bezier_top = np.array(BezierCoeff(t_plot)).dot(s1_bezier)
    Bezier_bottom = np.array(BezierCoeff(t_plot)).dot(s2_bezier)
    sampled_polygon = np.vstack((Bezier_top, Bezier_bottom))
    return sampled_polygon

def roll_pts(in_poly):
    # in_poly (np.array): (2 * sample_num_per_side, 2)
    num = in_poly.shape[0]
    assert num % 2 == 0
    return np.vstack((in_poly[num//2:], in_poly[:num//2])).reshape((-1)).tolist()

def intersec_num_y(polyline, x):
    '''
    Args:
        polyline: Represent the bottom side of a text instance
        x: Represent a vertical line.

    Returns:
        num: The intersection number of a vertical line and the polyline.
        ys_value: The y values of intersection points.

    '''
    num = 0
    ys_value = []
    for ip in range(7):
        now_x, now_y = polyline[ip][0], polyline[ip][1]
        next_x, next_y = polyline[ip+1][0], polyline[ip+1][1]
        if now_x == x:
            num += 1
            ys_value.append(now_y)
            continue
        xs, ys = [now_x, next_x], [now_y, next_y]
        min_xs, max_xs = min(xs), max(xs)
        if min_xs < x and max_xs > x:
            num += 1
            ys_value.append(((x-now_x)*(next_y-now_y)/(next_x-now_x)) + now_y)
    if polyline[7][0] == x:
        num += 1
        ys_value.append(polyline[7][1])
    assert len(ys_value) == num
    return num, ys_value

def process_polygon_positional_label_form(json_in, json_out):
    '''
    A simple implementation of generating the positional label 
    form for polygon points. There are still some special 
    situations need to be addressed, such as vertical instances 
    and instances in "C" shape. Maybe using a rotated box 
    proposal could be a better choice. If you want to generate 
    the positional label form for Bezier control points, you can 
    also firstly sample points on Bezier curves, then use the 
    on-curve points referring to this function to decide whether 
    to roll the original Bezier control points.

    (By the way, I deem that the "conflict" between point labels 
    in the original form also impacts the detector. For example, 
    in most cases, the first point appears in the upper left corner. 
    If an inverse instance turns up, the first point moves to the 
    lower right. Transformer decoders are supervised to address this 
    diagonal drift, which is like the noise pulse. It could make the 
    prediction unstable, especially for inverse-like instances. 
    This may be a limitation of control-point-based methods. 
    Segmentation-based methods are free from this issue. And there 
    is no need to consider the point order issue when using rotation 
    augmentation for segmentation-based methods.)

    Args:
        json_in: The path of the original annotation json file.
        json_out: The output json path.
    '''
    with open(json_in) as f_json_in:
        anno_dict = json.load(f_json_in)
    insts_list = anno_dict['annotations']
    new_insts_list = []
    roll_num = 0  # to count approximate inverse-like instances
    total_num = len(insts_list)
    for inst in tqdm(insts_list):
        new_inst = copy.deepcopy(inst)
        poly = np.array(inst['polys']).reshape((-1, 2))
        # suppose there are 16 points for each instance, 8 for each side
        assert poly.shape[0] == 16
        is_ccw = Polygon(poly).exterior.is_ccw
        # make all points in clockwise order
        if not is_ccw:
            poly = np.vstack((poly[8:][::-1, :], poly[:8][::-1, :]))
            assert poly.shape == (16,2)

        roll_flag = False
        start_line, end_line = poly[:8], poly[8:][::-1, :]

        if min(start_line[:, 1]) > max(end_line[:, 1]):
            roll_num += 1
            poly = roll_pts(poly)
            new_inst.update(polys=poly)
            new_insts_list.append(new_inst)
            continue

        # right and left
        if min(start_line[:, 0]) > max(end_line[:, 0]):
            if min(poly[:, 1]) == min(end_line[:, 1]):
                roll_flag = True
            if roll_flag:
                roll_num += 1
                poly = roll_pts(poly)
            if not isinstance(poly, list):
                poly = poly.reshape((-1)).tolist()
            new_inst.update(polys=poly)
            new_insts_list.append(new_inst)
            continue

        # left and right
        if max(start_line[:, 0]) < min(end_line[:, 0]):
            if min(poly[:, 1]) == min(end_line[:, 1]):
                roll_flag = True
            if roll_flag:
                roll_num += 1
                poly = roll_pts(poly)
            if not isinstance(poly, list):
                poly = poly.reshape((-1)).tolist()
            new_inst.update(polys=poly)
            new_insts_list.append(new_inst)
            continue

        for pt in start_line:
            x_value, y_value = pt[0], pt[1]
            intersec_with_end_line_num, intersec_with_end_line_ys = intersec_num_y(end_line, x_value)
            if intersec_with_end_line_num > 0:
                if max(intersec_with_end_line_ys) < y_value:
                    roll_flag = True
                    break
                if min(poly[:, 1]) == min(start_line[:, 1]):
                    roll_flag = False
                    break
        if roll_flag:
            roll_num += 1
            poly = roll_pts(poly)
            new_inst.update(polys=poly)
            new_insts_list.append(new_inst)
        else:
            if not isinstance(poly, list):
                poly = poly.reshape((-1)).tolist()
            new_inst.update(polys=poly)
            new_insts_list.append(new_inst)
    assert len(new_insts_list) == total_num

    anno_dict.update(annotations=new_insts_list)
    with open(json_out, mode='w+') as f_json_out:
        json.dump(anno_dict, f_json_out)

    # the approximate inverse-like ratio, the actual ratio should be lower
    print(f'Inverse-like Ratio: {roll_num / total_num * 100: .2f}%. Finished.')


if __name__ == '__main__':
    # an example of processing the positional label form for polygon control points.
    process_polygon_positional_label_form(
        json_in='./datasets/totaltext/train_poly_ori.json',
        json_out='./datasets/totaltext/train_poly_pos_example.json'
    )