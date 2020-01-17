# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Conversion between objects on image and target maps:
    ObjectInfo[s] <-> TargetMaps
"""
import itertools
from typing import List, Tuple

import cv2
import numpy as np

import utils
from data.data_info import ObjectInfo, TargetMapInfo


class Converter:
    """
    Class for converting
    object_infos <-> target_map
    """

    def __init__(self,
                 target_map_info: TargetMapInfo,
                 detection_pixel_threshold: float = 0.5,
                 detection_area_threshold: int = 10):
        """

        :param target_map_info:
        :param detection_pixel_threshold: threshold to binarize "detection" output channels
        :param detection_area_threshold: threshold to filter-out too small objects (with area < threshold)
        """
        self.target_map_info = target_map_info
        self.detection_pixel_threshold = detection_pixel_threshold
        self.detection_area_threshold = detection_area_threshold

    def build_target_map(self, object_infos: List[ObjectInfo], image_size: Tuple, for_visualization: bool = False):
        """
        object_infos -> target_map
        :param object_infos:
        :param image_size: (h, w)
        :param for_visualization
        :return:
        """
        # multi-class
        # 1)draw (h, w, CHANNELS) 0...1 mask
        # and that's it!
        return self._build_target_map(object_infos, image_size, for_visualization)

    def postprocess_target_map(self, target_map, merge=True):
        """
        target_map -> object_infos
        :param target_map: numpy array of shape (B, C, H, W)
        :param merge:
        :return:
            1)if merge=False -
                Dict[str, List[List[ObjectInfo]]]
                detection_class_name ->
                    list (for each image in batch) of lists (detected objects of that class for that image)
            2)if merge=True -
                List[List[ObjectInfo]]
                list (for each image in batch) of lists (detected objects REGARDLESS their class for that image)
        """
        tm_size = target_map.shape[-2:]

        detected_objects = dict()  # d_name -> detected objects
        for d_class_id, d_tm_id, d_classified, d_name in zip(
                self.target_map_info.d_class_ids,
                self.target_map_info.d_tm_ids,
                self.target_map_info.d_classified,
                self.target_map_info.d_class_names
        ):
            d_binarized_map = (target_map[:, d_tm_id] > self.detection_pixel_threshold).astype(np.uint8)
            if not d_classified:
                d_detected_objects = [
                    [ObjectInfo(location=box, class_name=d_name, class_id=(d_class_id, None))
                     for box in utils.get_contours_and_boxes(d_binarized_map[i],
                                                             min_area=self.detection_area_threshold)[1]]
                    for i in range(d_binarized_map.shape[0])
                ]
                # saving all detected objects for this *d*
                detected_objects[d_name] = d_detected_objects
                continue  # moving on to the next detection class

            # if d is classified
            d_detected_objects = []
            for i in range(d_binarized_map.shape[0]):
                # get locations
                contours, boxes = utils.get_contours_and_boxes(
                    binarized_map=d_binarized_map[i], min_area=self.detection_area_threshold)
                # get classes
                img_detected_objects = []
                for cnt, box in zip(contours, boxes):
                    c_class_ids = self.target_map_info.get_c_class_ids(d_class_id)
                    c_tm_ids = self.target_map_info.get_c_tm_ids(d_class_id)
                    mask = self._get_objects_mask(image_size=tm_size, objects=[self._round_contour(box)])
                    class_pixel_scores = target_map[i, c_tm_ids][:, mask.astype(np.bool)]
                    class_avg_scores = np.mean(class_pixel_scores, axis=1)
                    best_class_idx = np.argmax(class_avg_scores)

                    c_class_id = c_class_ids[best_class_idx]
                    c_class_name = self.target_map_info.get_c_names(d_class_id)[best_class_idx]

                    img_detected_objects.append(
                        ObjectInfo(location=box,
                                   class_name=c_class_name,
                                   class_id=(d_class_id, c_class_id))
                    )
                d_detected_objects.append(img_detected_objects)

            # saving all detected objects for this *d*
            detected_objects[d_name] = d_detected_objects

        if merge:
            detected_objects = [  # image_idx -> all objects list
                [o for d_image_objects in detected_objects.values() for o in d_image_objects[i]]
                for i in range(target_map.shape[0])
            ]
        return detected_objects

    @staticmethod
    def _get_objects_mask(image_size: Tuple, objects: List[np.ndarray], thickness=-1, mask=None):
        if mask is None:
            mask = np.zeros(image_size, dtype=np.uint8)
        else:
            assert mask.shape == image_size and mask.dtype == np.uint8
        cv2.drawContours(mask, objects, -1, 1, thickness=thickness)
        return mask

    @staticmethod
    def _round_contour(contour, **kwargs):
        return np.round(contour).astype(np.int32)

    @staticmethod
    def _build_binary_map(object_infos: List[ObjectInfo], image_size: Tuple, mask_dim: int = 2):
        # object_infos supposed to be filtered to contain single class!
        drawn_contours = [Converter._round_contour(obj.location) for obj in object_infos]
        mask = Converter._get_objects_mask(image_size, objects=drawn_contours)
        if mask_dim == 3:
            mask = np.expand_dims(mask, -1)
        return mask

    def _build_target_map(self, object_infos: List[ObjectInfo], image_size: Tuple, for_visualization: bool = False):
        drawn_contours = sorted([
            (
                Converter._round_contour(obj.location),
                self.target_map_info.get_tm_ids(obj.d_class_id, obj.c_class_id)
            ) for obj in object_infos
        ], key=lambda pair: pair[1])

        if not for_visualization:
            mask_shape = (self.target_map_info.n_channels,) + image_size
        else:
            mask_shape = image_size
        mask = np.zeros(mask_shape, dtype=np.uint8)

        for (d_tm_id, c_tm_id), group in itertools.groupby(drawn_contours, key=lambda pair: pair[1]):
            curr_contours = [pair[0] for pair in group]
            if not for_visualization:
                Converter._get_objects_mask(image_size, curr_contours, mask=mask[d_tm_id])
                if c_tm_id is not None:
                    Converter._get_objects_mask(image_size, curr_contours, mask=mask[c_tm_id])
            else:
                drawn_mask = Converter._get_objects_mask(image_size, curr_contours)
                draw_color = d_tm_id + 1 if c_tm_id is None else c_tm_id
                mask[drawn_mask.astype(np.bool)] = draw_color

        if not for_visualization:
            mask = mask.transpose((1, 2, 0))
        return mask
