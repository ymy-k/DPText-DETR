import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
import sys

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing, Point, LineString
from adet.evaluation import text_eval_script_det
import zipfile
import pickle


# Modified from TESTR. Only the detection metrics are evaluated.
class TextDetEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )

        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # For ICDAR ArT2019 evaluation on the official website.
        # The saved json file can be directly submitted to the website.
        self.submit = False

        # use dataset_name to decide eval_gt_path
        if "rotate" in dataset_name:
            if "totaltext" in dataset_name:
                self._text_eval_gt_path = "datasets/evaluation/gt_totaltext_rotate.zip"
        elif "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_totaltext.zip"
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_ctw1500.zip"
        elif "art" in dataset_name:
            self._text_eval_gt_path = None
            self.submit = True
        elif "inversetext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_inversetext.zip"
        else:
            raise NotImplementedError

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "file_name": input["file_name"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input["image_id"], input["file_name"])
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results"):

        def compute_area(polys):
            poly = copy.deepcopy(polys)
            return Polygon(np.array(poly).reshape((-1,2))).area

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    poly_area = compute_area(data[ix]['polys'])
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])  # 'image_id'
                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + str(int(data[ix]['polys'][i][0])) +','+str(int(data[ix]['polys'][i][1])) +','
                        outstr = outstr + str(round(data[ix]['score'], 3)) + ',' + '####' + '\n'
                        f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        if not os.path.isdir(dirn):
            os.mkdir(dirn)

        for line in fres:
            line = line.strip()
            s = line.split(': ')
            filename = '{:07d}.txt'.format(int(s[0]))
            outName = os.path.join(dirn, filename)
            with open(outName, 'a') as fout:
                ptr = s[1].strip().split(',')
                assert ptr[-1] == '####'
                score = ptr[-2]
                cors = ','.join(e for e in ptr[:-2])
                fout.writelines(cors+',####'+'\n')
        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_"+temp_dir

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file+'*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',')
                cors = ptr[:-1]
                assert(len(cors) %2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                
                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                    
                pRing = LinearRing(pts)
                if pRing.is_ccw:
                    pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                outstr = outstr+',####'
                fout.writelines(outstr+'\n')
            fout.close()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir("../")
        # clean temp files
        shutil.rmtree(origin_file)
        shutil.rmtree(output_file)
        return "det.zip"
    
    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script_det.text_eval_main_det(det_file=result_path, gt_file=gt_path)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}
        PathManager.mkdirs(self._output_dir)

        if self.submit:
            file_path = os.path.join(self._output_dir, "art_submit.json")
            coco_results = {}
            for prediction in predictions:
                key = 'res_' + prediction['file_name'].split('/')[-1].split('.')[0].split('_')[-1]
                coco_results[key] = prediction["instances"]
        else:
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
            file_path = os.path.join(self._output_dir, "text_results.json")

        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()
        
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        # eval text
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir)
        result_path = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
        os.remove(result_path)

        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        for task in ["det_method"]:
            result = text_result[task]
            groups = re.match(template, result).groups()
            self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}

        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, img_id, img_name):
        img_name = img_name.split('/')[-1].split('.')[0]
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        if self.use_polygon:
            pnts = instances.polygons.numpy()
        else:
            pnts = instances.beziers.numpy()
    
        results = []
        if self.submit:
            for pnt, score in zip(pnts, scores):
                poly = self.pnt_to_polygon(pnt)  # list
                poly = [(int(p[0]), int(p[1])) for p in poly]
                assert(len(poly) %2 == 0 and len(poly) >= 3), 'cors invalid.'
                try:
                    pgt = Polygon(poly)
                except Exception as e:
                    continue
                if not pgt.is_valid:
                    continue

                is_ccw = pgt.exterior.is_ccw
                if not is_ccw:
                    poly = poly[::-1]

                result = {
                    "points": poly,
                    "confidence": score
                }
                results.append(result)
            return results
        else:
            for pnt, score in zip(pnts, scores):
                poly = self.pnt_to_polygon(pnt)
                result = {
                    "image_id": img_id,
                    "category_id": 1,
                    "polys": poly,
                    "score": score,
                    "image_name": img_name,
                }
                results.append(result)
            return results


    def pnt_to_polygon(self, ctrl_pnt):
        if self.use_polygon:
            return ctrl_pnt.reshape(-1, 2).tolist()
        else:
            u = np.linspace(0, 1, 20)
            ctrl_pnt = ctrl_pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, ctrl_pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), ctrl_pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), ctrl_pnt[:, 2]) \
                + np.outer(u ** 3, ctrl_pnt[:, 3])
            
            # convert points to polygon
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
            return points.tolist()