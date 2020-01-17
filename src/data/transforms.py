# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Albumentations custom transforms and transformation sequences
"""
import albumentations as A
import cv2
import numpy as np
from albumentations import (
    ChannelShuffle, CLAHE, Compose, HueSaturationValue, IAAPerspective,
    JpegCompression, Normalize, OneOf, PadIfNeeded,
    RandomBrightnessContrast, RandomGamma, RGBShift, ShiftScaleRotate, ToGray,
    DualTransform, BasicTransform
)
from albumentations.augmentations import functional as F
from catalyst.utils import tensor_from_rgb_image

import utils
from converter import Converter

# concurrent stability
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class LongestMaxSizeMinMultiple(DualTransform):
    """
    Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
    This transformation also adds `x_scale` and `y_scale` into transformation dict
    """

    def __init__(self, max_size=1024, min_multiple=4,
                 interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1):
        """
        :param max_size (int): maximum size of the image after the transformation
        :param min_multiple (int): minimal multiple for new height and width
        :param interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR
        :param always_apply (bool):
        :param p (float): probability of applying the transform. Default: 1
        """
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        assert max_size % min_multiple == 0, "must be evenly divisible"
        self.max_size = max_size
        self.min_multiple = min_multiple

    def apply_with_params(self, params, force_apply=False, **kwargs):
        result = super().apply_with_params(params, force_apply, **kwargs)
        # add scales to transformed dict
        for key in ("x_scale", "y_scale"):
            if key in result:
                result[key] *= params[key]
            else:
                result[key] = params[key]
        return result

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]
        if max(rows, cols) <= self.max_size:
            # just round each appropriately
            new_rows = utils.get_closest_divisible(rows, self.min_multiple)
            new_cols = utils.get_closest_divisible(cols, self.min_multiple)
        elif rows > cols:
            new_rows = self.max_size
            new_cols = utils.get_closest_divisible(cols * (new_rows / rows), self.min_multiple)
        else:
            # rows <= cols
            new_cols = self.max_size
            new_rows = utils.get_closest_divisible(rows * (new_cols / cols), self.min_multiple)
        x_scale = new_cols / cols
        y_scale = new_rows / rows
        params.update({"x_scale": x_scale, "y_scale": y_scale, "new_rows": new_rows, "new_cols": new_cols})
        return params

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(img, height=params["new_rows"], width=params["new_cols"], interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        raise NotImplementedError("We may have changed aspect ratio, so bboxes must be transformed appropriately")

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_scale(keypoint, params["x_scale"], params["y_scale"])

    def get_transform_init_args_names(self):
        return ("max_size", "min_multiple", "interpolation")


class ToTensor(A.core.transforms_interface.DualTransform):
    """Convert image and mask to ``torch.Tensor``"""

    def __call__(self, force_apply: bool = True, **kwargs):
        """Convert image and mask to ``torch.Tensor``"""
        kwargs.update(image=tensor_from_rgb_image(kwargs["image"]))
        if "mask" in kwargs.keys():
            kwargs.update(mask=tensor_from_rgb_image(kwargs["mask"]).float())

        return kwargs


class ImageToRGB(A.core.transforms_interface.ImageOnlyTransform):
    """Convert the input image to RGB if it grayscale"""

    def __init__(self, always_apply: bool = False, p: float = 1.0):
        """
        Args:
            p (float): probability of applying the transform
        """
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Convert the input image to RGB if it grayscale"""
        if len(img.shape) < 3 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


def _add_transform_default_params(kwargs):
    if "keypoint_params" not in kwargs:
        kwargs["keypoint_params"] = {'format': 'xy', 'remove_invisible': False}
    return kwargs


def pre_transforms(image_size: int = 256, min_multiple: int = 64, **kwargs):
    """Transforms that always be applied before other transformations"""
    _add_transform_default_params(kwargs)

    transforms = Compose([
        ImageToRGB(),
        LongestMaxSizeMinMultiple(max_size=image_size, min_multiple=min_multiple),
    ], **kwargs)
    return transforms


def post_transforms():
    """Transforms that always be applied after all other transformations"""
    return Compose([Normalize(), ToTensor()])


def hard_transform(image_size: int = 256, p: float = 0.5, **kwargs):
    """Hard augmentations (on training)"""
    _add_transform_default_params(kwargs)

    transforms = Compose([
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=p,
        ),
        IAAPerspective(scale=(0.02, 0.05), p=p),
        OneOf([
            HueSaturationValue(p=p),
            ToGray(p=p),
            RGBShift(p=p),
            ChannelShuffle(p=p),
        ]),
        RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, p=p
        ),
        RandomGamma(p=p),
        CLAHE(p=p),
        JpegCompression(quality_lower=50, p=p),
        PadIfNeeded(
            image_size, image_size, border_mode=cv2.BORDER_CONSTANT
        ),
    ], **kwargs)
    return transforms


def soft_transform(image_size: int = 256, p: float = 0.5, **kwargs):
    """Soft trainsforms (on validation)"""
    _add_transform_default_params(kwargs)

    transforms = Compose([
        PadIfNeeded(
            image_size, image_size, border_mode=cv2.BORDER_CONSTANT
        ),
    ], **kwargs)
    return transforms


class DictTransformer:
    """
    Class for transforming input dict with image, mask, objects, etc
    Should be used as `transform` function for each torch dataset
    Has __call__ method to do all transforms
    """

    def __init__(self, converter: Converter, transform_fn: BasicTransform, build_before=False):
        """
        :param converter: converter between target maps and objects
        :param transform_fn: albumentations transform function
        :param build_before: if True - build target map and then augment, if False - augment, then build target map
                        build_before may have impact on performance speed (False is faster if mask is not precomputed)
        """
        self.converter = converter
        self.transform_fn = transform_fn
        if build_before:
            self._call = self._call_build_before
        else:
            self._call = self._call_build_after

    def __call__(self, dict_):
        return self._call(dict_)

    def _call_build_before(self, dict_):
        if "mask" not in dict_:
            assert "objects" in dict_
            dict_["mask"] = self.converter.build_target_map(dict_["objects"], dict_["image"].shape[:2])

        result = self.transform_fn(**dict_)
        return result

    def _call_build_after(self, dict_):
        assert "objects" in dict_
        dict_["keypoints"] = utils.objects_to_keypoint(dict_["objects"])

        result = self.transform_fn(**dict_)
        result["objects_on_original"] = result["objects"]
        result["objects"] = utils.keypoints_to_objs(result["keypoints"], dict_["objects"])
        if "mask" not in dict_:
            result["mask"] = self.converter.build_target_map(result["objects"], result["image"].size()[1:])
            result["mask"] = tensor_from_rgb_image(result["mask"]).float()
        return result
