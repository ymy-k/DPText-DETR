import random
import numpy as np
from fvcore.transforms import transform as T
from detectron2.data.transforms import RandomCrop, StandardAugInput
from detectron2.structures import BoxMode
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform
import albumentations as A


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class BlurTransform(Transform):
    def __init__(self, kernel_size, p):
        super().__init__()
        blur_aug = A.OneOf([
            A.Blur(blur_limit=kernel_size, p=1),
            A.MotionBlur(blur_limit=kernel_size, p=1)
        ], p=p)
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.blur_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomBlur(Augmentation):
    def __init__(self, kernel_size, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return BlurTransform(self.kernel_size, self.p)


class GaussNoiseTransform(Transform):
    def __init__(self, p):
        super().__init__()
        gauss_noise_aug = A.GaussNoise(p=p)
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.gauss_noise_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class GaussNoise(Augmentation):
    def __init__(self, p):
        """
        Args:
            p (float): probability
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return GaussNoiseTransform(self.p)


class HueSaturationValueTransform(Transform):
    def __init__(self, hue_shift_limit, p):
        super().__init__()
        hue_saturation_aug = A.HueSaturationValue(hue_shift_limit=hue_shift_limit, p=p)
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        return self.hue_saturation_aug(image=img)['image']

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomHueSaturationValue(Augmentation):
    """
    Random hue, saturation & value.
    """
    def __init__(self, hue_shift_limit, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return HueSaturationValueTransform(self.hue_shift_limit, self.p)