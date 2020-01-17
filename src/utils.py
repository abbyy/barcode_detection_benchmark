# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Various helper functions
"""

import itertools
import logging
import os
import platform
from multiprocessing import current_process
from typing import Dict

import cv2
import numpy as np
from catalyst.utils import merge_dicts
from shapely.geometry import Polygon, MultiPoint


def is_main_process():
    """:returns True if function is executed in MainProcess, False otherwise"""
    return current_process().name == 'MainProcess'


def is_os_windows():
    """:returns True if the current OS is Windows of any kind"""
    return platform.system().lower().startswith("windows")


def read_image(filename, flags=None, grey=False) -> np.ndarray:
    """
    Reads image from file as numpy array
    :param filename: image filename
    :param flags: opencv flags (may be useful to set cv2.IGNORE_ORIENTATION to read our markup properly)
    :param grey: if True return grey image, otherwise 3-channel RGB
    :return:
    """
    image = cv2.imread(filename, flags=flags)
    if image is None:
        raise ValueError("Can't read image '{filename}', opencv return None")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if grey else cv2.COLOR_BGR2RGB)


def has_image_extension(filename) -> bool:
    """
    Checks if filename has image extension
    :param filename: filename to check
    :return: True if file has image extension, False otherwise
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}


def has_xml_extension(filename) -> bool:
    """
    Checks if filename has image extension
    :param filename: filename to check
    :return: True if file has image extension, False otherwise
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.xml'


def find_corresponding_markup_filename(image_filename, markup_dir, ext='.xml'):
    """
    Finds markup filename with same, if markup file not found raises FileNotFoundError
    :param image_filename:
    :param markup_dir:
    :param ext: markup file extension
    :return:
    """
    markup_filename = os.path.join(markup_dir, os.path.splitext(image_filename)[0] + ext)
    if not os.path.exists(markup_filename):
        raise FileNotFoundError(f"Can't find markup for image '{image_filename}' "
                                f"in markup dir '{markup_dir}' with extension '{ext}'")
    return markup_filename


def fix_polygon(polygon):
    """
    Transforms list of points into their convex hull
    :param polygon: list of N 2D points coordinates [[x1, y1], [x2, y2], ..., [xN, yN]]
    :return: convex hull in the same format as input [[x_c1, y_c1], ..., [x_cM, y_cM]]
    """
    polygon = np.array(polygon)
    poly = Polygon(polygon)
    is_valid = poly.is_valid
    if not is_valid:
        logging.info('polygon invalid')
        fixed_poly = MultiPoint(polygon).convex_hull
        fixed_poly = np.array(fixed_poly.exterior.coords)
        return fixed_poly
    return polygon


def objects_to_keypoint(object_infos):
    return [
        (xy[0], xy[1], instance_id)
        for instance_id, obj_info in enumerate(object_infos)
        for xy in obj_info.location
    ]


def keypoints_to_objs(keypoints, object_infos):
    keypoints = sorted(keypoints, key=lambda kp: kp[2])  # sort by instance id
    objs = []
    for instance_id, group in itertools.groupby(keypoints, key=lambda kp: kp[2]):
        instance_keypoints = [(xy_id[0], xy_id[1]) for xy_id in group]
        poly = fix_polygon(instance_keypoints)
        objs.append(
            object_infos[instance_id].create_same_class_with_different_location(new_location=poly)
        )
    return objs


def extract_locations_and_object_types(
        object_info,
        classification=False,
        object_types_format="name"
):
    """

    :param object_info:
    :param classification:
    :param object_types_format:
    :return:
    """

    locations = [obj.location for obj in object_info]

    if classification:
        if object_types_format == "id":
            obj_types = [(obj.d_class_id, obj.c_class_id) for obj in object_info]
        elif object_types_format == "name":
            obj_types = [obj.class_name for obj in object_info]
        else:
            raise ValueError("Unsupported object type format")
    else:
        obj_types = None
    return locations, obj_types


def get_contours_and_boxes(binarized_map, min_area=0):
    """

    :param binarized_map: np.array of np.uint8
    :param min_area:
    :return:
    """
    assert binarized_map.dtype == np.uint8
    contours, _ = cv2.findContours(
        binarized_map,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > min_area, contours))
    rects = [cv2.minAreaRect(cnt) for cnt in contours]
    boxes = [cv2.boxPoints(rect).reshape((-1, 2)) for rect in rects]
    assert len(boxes) == len(contours)

    return contours, boxes


def process_class_config(class_config: Dict):
    default_w = 1.0
    default_classification_w = 0.05
    default_subclasses = dict()
    default_params = {
        "weight": default_w,
        "classification_w": default_classification_w,
        "subclasses": default_subclasses,
        "classified": True
    }

    for d_class_name in class_config.keys():
        class_config[d_class_name] = merge_dicts(
            default_params,
            class_config.get(d_class_name) or {}
        )
        for c_class_name in class_config[d_class_name]["subclasses"]:
            class_config[d_class_name]["subclasses"][c_class_name] = merge_dicts(
                {"weight": default_w, "aliases": [c_class_name]},
                class_config[d_class_name]["subclasses"][c_class_name] or {}
            )

    return class_config


def rescale_objects(objects, x_scale, y_scale):
    return [
        o.create_same_class_with_different_location(
            o.location * np.array([[x_scale, y_scale]])
        ) for o in objects
    ]


def get_closest_divisible(x, y):
    """returns closest value to `x` evenly divisible by `y`; the result is always greater than 0"""
    assert x > 0 and y > 0
    assert isinstance(y, int)
    if x < y:
        return y
    return int(round(x / y)) * y
