# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Readers for each field of the dataset
"""
import logging
import os
import xml.etree.ElementTree
from typing import Dict

import cv2
from catalyst.data import ImageReader, ReaderSpec

import utils
from data.data_info import ObjectInfo


class FineImageReader(ImageReader):
    """
    Image reader implementation which can read any extension (.tiff support)
    """

    def __call__(self, row):
        """Reads a row from your annotations dict with filename and
        transfer it to an image

        Args:
            row: elem in your dataset.

        Returns:
            np.ndarray: Image
        """
        image_name = str(row[self.input_key])
        img = cv2.imread(
            os.path.join(self.datapath, image_name)
        )
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = {self.output_key: img}
        return result


class XMLBarcodeObjectsReader(ReaderSpec):
    """Reader for barcode objects from .xml
    """

    def __init__(self, input_key: str, output_key: str, datapath: str = None,
                 type_conversion_rules: Dict = None, unk_type_str: str = None,
                 markup_name2id: Dict = None, markup_name2class_name: Dict = None):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            datapath (str): path to images dataset
                (so your can use relative paths in annotations)
            type_conversion_rules (Dict): dict which maps `in_type_name` to `out_type_name`
            unk_type_str (str): special key for `type_conversion_rules` dict,
                which, if present in the dict, is applied to all object_types which are not in the dict
            markup_name2id (Dict): markup type name -> object id
            markup_name2class_name (Dict): markup name to class name
        """
        super().__init__(input_key, output_key)
        self.datapath = datapath
        self.type_conversion_rules = type_conversion_rules
        self.unk_type_str = unk_type_str
        self.markup_name2id = markup_name2id
        self.markup_name2class_name = markup_name2class_name

        self._ready_objects = dict()  # cached result

    def __call__(self, row):
        """Reads a row from your annotations dict and
        transfer it to data, needed by your network
        for example open image by path, or read string and tokenize it.

        Args:
            row: elem in your dataset.

        Returns:
            Data object used for your neural network
        """
        objects_fname = str(row[self.input_key])
        objects = self._get_objects(objects_fname)

        result = {self.output_key: objects}
        return result

    def _get_objects(self, objects_fname):
        if objects_fname not in self._ready_objects:
            objects = self._read_markup_from_file(os.path.join(self.datapath, objects_fname), skip_empty_markup=True)
            self._ready_objects[objects_fname] = objects
        return self._ready_objects[objects_fname]

    def _read_markup_from_file(self, markup_file_path, skip_empty_markup=True):
        """
        Read markup file to get objects on the image
        :param markup_file_path:
        :param skip_empty_markup: raise error if no objects are specified in markup file
        :return: list of ObjectInfo
        """
        markup = []
        words = xml.etree.ElementTree.parse(markup_file_path).getiterator("Barcode")
        for word in words:
            border_points = []
            barcode_type = word.get('Type')
            if not barcode_type or barcode_type.strip() == "":
                # this may be a valid barcode which we want to recognize
                # in that case throwing it out may be a bad idea
                # and it may also be invalid barcode which we want to throw out
                raise ValueError("Unknown barcode type (empty string). "
                                 "Image {} will be skipped".format(os.path.basename(markup_file_path)))
            before_conversion_type = barcode_type
            barcode_type = self._convert_object_type(barcode_type)
            if barcode_type is None:
                logging.info(f"skipping object which type is converted to None: '{before_conversion_type}'")
                continue

            for point in word.iter('Point'):
                border_points.append(
                    (float(point.get('X')), float(point.get('Y')))
                )
            assert len(border_points) == 4

            quad_vertices = utils.fix_polygon(border_points)
            assert quad_vertices.shape == (4, 2)

            markup.append(ObjectInfo(location=quad_vertices,
                                     class_name=self.markup_name2class_name[barcode_type],
                                     class_id=self.markup_name2id[barcode_type]))

        if not markup and skip_empty_markup:
            raise ValueError(
                "Skipping suspicious empty markup file (no barcodes in markup) {}".format(markup_file_path))

        return markup

    def _convert_object_type(self, in_type):
        """
        Converts input object type according to the rules
        :param in_type:
        :return: out_type
        """
        if self.type_conversion_rules is not None:
            if in_type in self.type_conversion_rules:
                in_type = self.type_conversion_rules[in_type]
            elif self.unk_type_str is None:
                return None
            else:
                in_type = self.type_conversion_rules[self.unk_type_str]
        if in_type not in self.markup_name2class_name:
            if self.unk_type_str is None or self.unk_type_str not in self.markup_name2class_name:
                return None
        return in_type


object_file_readers = {
    'barcode_xml': XMLBarcodeObjectsReader
}
