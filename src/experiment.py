# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Catalyst experiment classes
"""
import collections
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List

import pandas as pd
from catalyst.data import (
    ListDataset, ReaderCompose, ScalarReader
)
from catalyst.dl import ConfigExperiment
from catalyst.dl.utils import set_global_seed
from catalyst.utils.pandas import dataframe_to_list
from torch.utils.data import DataLoader

from converter import Converter
from data.batch_collate import get_collate_fn
from data.data_info import TargetMapInfo
from data.data_readers import FineImageReader, object_file_readers
from data.datasets import CachedDataset
from data.transforms import pre_transforms, post_transforms, hard_transform, soft_transform, Compose, DictTransformer
from utils import process_class_config


class _Modes:
    """
    Common modes (just constants to avoid misprints)
    """
    TRAIN = "train"
    VALID = "valid"
    INFER = "infer"


class SimpleExperiment(ConfigExperiment):

    def __init__(self, config: Dict):
        super().__init__(config)
        self._class_config = config.get("class_config", None)
        assert self._class_config is not None, "class_config must be specified in yml"
        self._class_config = process_class_config(self._class_config)
        self.tm_info = TargetMapInfo(self._class_config)

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None,
                       image_size: int = 256, min_multiple: int = 64,
                       detection_pixel_threshold: float = 0.5,
                       detection_area_threshold: int = 10):
        """

        :param stage:
        :param mode:
        :param image_size: maximal image side for transformed image
        :param min_multiple: minimal multiple for each image side of transformed image
            (it is recommended to use your neural network scale here)
        :param detection_pixel_threshold: threshold to binarize "detection" output channels
        :param detection_area_threshold: threshold to filter-out too small objects (with area < threshold)
        :return:
        """
        pre_transform_fn = pre_transforms(image_size=image_size, min_multiple=min_multiple)

        if mode == _Modes.TRAIN:
            post_transform_fn = Compose([
                hard_transform(image_size=image_size),
                post_transforms(),
            ])
        elif mode == _Modes.VALID:
            post_transform_fn = Compose([
                soft_transform(image_size=image_size),
                post_transforms()
            ])
        elif mode == _Modes.INFER:
            post_transform_fn = post_transforms()
        else:
            raise NotImplementedError()

        transform_fn = Compose([pre_transform_fn, post_transform_fn])
        converter = Converter(TargetMapInfo())

        process = DictTransformer(
            converter=converter,
            transform_fn=transform_fn,
            build_before=False
        )

        return process

    def get_datasets(
            self,
            stage: str,
            datasets_params: List[Dict] = None,
            # transforms params
            image_size: int = 256,
            min_multiple: int = 64,
            detection_pixel_threshold: float = 0.5,
            detection_area_threshold: int = 10,
            #
            max_caching_size: int = 1000,
            type_conversion_rules: Dict = None,
            **kwargs
    ):
        datasets = collections.OrderedDict()
        for dataset_name, dataset_params in datasets_params.items():
            # `mode` on `"infer"` stage is always `"infer"`
            # on train stage remove inference dataset
            # on infer stage mask train dataset as valid_*
            mode = _Modes.TRAIN
            if stage.startswith(_Modes.INFER):
                mode = _Modes.INFER
                if dataset_name.startswith(_Modes.TRAIN):
                    # mask train dataset on infer stage
                    dataset_name = f"valid_{dataset_name}"
            elif dataset_name.startswith(_Modes.VALID):
                mode = _Modes.VALID
            elif dataset_name.startswith(_Modes.INFER):
                mode = _Modes.INFER

            if not stage.startswith(_Modes.INFER) and dataset_name.startswith(_Modes.INFER):
                continue  # skip infer dataset on training
            if stage.startswith(_Modes.INFER) and not dataset_name.startswith(_Modes.INFER):
                continue  # skip train/valid on inference

            dataset = self.get_dataset(stage=stage, mode=mode,
                                       image_size=image_size,
                                       min_multiple=min_multiple,
                                       detection_area_threshold=detection_area_threshold,
                                       detection_pixel_threshold=detection_pixel_threshold,
                                       **dataset_params)
            if len(dataset) > 0:
                if mode == _Modes.VALID and len(dataset) <= max_caching_size:
                    logging.info(f"Caching validation dataset '{dataset_name}'")
                    dataset = CachedDataset(dataset)
                datasets[dataset_name] = dataset

        return datasets

    def get_dataset(self, datapath, csv_path, info_json_path,
                    stage, mode, image_size, min_multiple,
                    detection_pixel_threshold,
                    detection_area_threshold):
        df_as_list = dataframe_to_list(pd.read_csv(csv_path, sep=','))
        with open(info_json_path, "r") as json_file:
            dataset_info = json.load(json_file)
        return ListDataset(
            list_data=df_as_list,
            open_fn=self.get_open_fn(dataset_info, datapath),
            dict_transform=self.get_transforms(
                stage=stage, mode=mode, image_size=image_size, min_multiple=min_multiple,
                detection_pixel_threshold=detection_pixel_threshold,
                detection_area_threshold=detection_area_threshold
            )
        )

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_params = self.stages_config[stage]["data_params"]
        loader_params_key = "loaders_params"
        collate_fn_key = "collate_fn"
        worker_init_fn_key = "worker_init_fn"

        if loader_params_key in data_params:
            loader_params = data_params[loader_params_key]
            assert isinstance(loader_params, dict), \
                f"{loader_params} should be Dict"
            # loader_params: ("train": ..., "valid": ..., ...) key==dataset, value==this_dataset_params
            for dataset_name, loader_overriten_params in loader_params.items():
                if collate_fn_key in loader_overriten_params:
                    loader_overriten_params[collate_fn_key] = get_collate_fn(
                        loader_overriten_params[collate_fn_key]
                    )
                if worker_init_fn_key not in loader_overriten_params:
                    loader_overriten_params[worker_init_fn_key] = self._worker_init_fn

        return super().get_loaders(stage)

    def _worker_init_fn(self, x):
        # can not be lambda if we want to run num_workers > 0 on windows
        set_global_seed(self.initial_seed + x)

    def get_open_fn(self, dataset_info, datapath, datapath_prefix=None):
        if datapath_prefix is not None:
            datapath = os.path.join(datapath, datapath_prefix)
        open_fn = ReaderCompose(readers=[
            FineImageReader(
                input_key="image", output_key="image", datapath=datapath
            ),
            object_file_readers[dataset_info["objects_format"]](
                input_key="objects", output_key="objects", datapath=datapath,
                markup_name2class_name=self.tm_info.alias2name,
                markup_name2id=self.tm_info.alias2dc
            ),
            ScalarReader(
                input_key="ID",
                output_key="image_name",
                default_value=-1,
                dtype=str,
            ),
        ])
        return open_fn
