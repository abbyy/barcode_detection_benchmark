# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
ObjectInfo & TargetMapInfo
"""
from typing import Any

import numpy as np

from utils import is_main_process, is_os_windows


class ObjectInfo:
    """
    Information about single object (location and class)
    """
    __slots__ = ['location', 'class_name', 'd_class_id', 'c_class_id']

    def __init__(self, location: np.ndarray, class_name: str, class_id: (int, int)):
        """

        :param location: np.array of shape (N, 2)
        :param class_name:
        :param class_id: (d_class_id, c_class_id) where c_class_id may be None
        """
        self.location = location
        self.class_name = class_name
        self.d_class_id = class_id[0]
        self.c_class_id = class_id[1]

    def create_same_class_with_different_location(self, new_location):
        return ObjectInfo(location=new_location, class_name=self.class_name,
                          class_id=(self.d_class_id, self.c_class_id))

    def __repr__(self) -> str:
        return f"Object(class_name={self.class_name}, location={self.location})"


class TargetMapInfo:
    """
    The most important class which contains information about network structure, including:
    - d_class_names
    - d_class_ids
    - d_target_map_ids  (channels on target map for d_class_ids)
    - d_classification_weights  (loss weights for classification of particular d_class)
    - d_class_weights  (loss weights for d_classes)
    - d_classified  (is d_class_id classified?)

    - c_class_names
    - c_class_ids  (id of class inside class)
    - c_target_map_ids  (channels on target map for c_class_ids

    - c_class_weights  (loss weights for c_classes) it is classification weight for specific d_class in loss
        the overall loss is supposed to be computed as follows (pseudo code):
            for d_class_id in d_class_ids:
                detection_loss = mean(detection_losses_d_class_id)
                if d_classified[d_class_id]:
                    c_loss[d_class_id] = \
                        weighted_mean(c_losses[c_class_ids[d_class_id]],
                                      weights=c_class_weights[c_class_ids[d_class_id]])

                d_total_losses[d_class_id] = (
                    1.0 * d_loss[d_class_id]
                    + d_classification_weights[d_class_id] * c_loss[d_class_id]
                ) / (1.0 + d_classification_weights[d_class_id])
            total_loss = weighted_mean(d_total_losses[d_class_id], weights=d_class_weights)
    """
    _instance = None

    def __new__(cls, config=None) -> Any:
        # this is a SUPER TRICKY HACK to work with singleton in multiprocessing (enable num_workers > 0 on windows)
        # it exploits the IMPL DETAIL - this class will not be constructed in the spawn threads
        # (the multiprocessing is used only in pytorch dataloader which do not create such object)
        # if you know the less terrible and devastating way to write this, please share
        if config is None and not is_main_process() and is_os_windows():
            return object.__new__(cls)  # we do not need this object (from IMPL DETAIL)

        # this is a singleton - once initialized (in the __init__ of experiment) it is available from everywhere
        if cls._instance is None:
            assert config is not None, "maybe you used relative module import of this file? don't do this"
            cls._instance = object.__new__(cls)
        else:
            assert config is None
        return cls._instance

    def __init__(self, config=None):
        """
        :param config (dict): detection_class_name -> classification_class_names -> ...
        """
        if config is None:
            return
        self._d_names = []
        self._d_weights = []
        self._d_classification_weights = []
        self._d_classified = []

        # d_class_id -> c_class_id -> {target_value}
        self._c_names = dict()  # d_class_id -> list of class names
        self._c_ids = dict()  # d_class_id -> array of ids
        self._c_tm_ids = dict()  # d_class_id -> array of (classification) target map ids
        self._c_weights = dict()  # d_class_id -> array of weights

        self._alias2dc = dict()  # name in markup -> (d_class_id, c_class_id)
        self._alias2name = dict()  # name in markup -> class name
        self._class2tm_id = dict()  # (d_class_id, c_class_id) -> (d_tm_id, c_tm_id)

        self._init_from_config(config)

    def _init_from_config(self, config):
        n_d_classes = len(config.keys())
        n_c_classes = 0

        for d_id, (d_class_name, d_class_params) in enumerate(config.items()):
            self._d_names.append(d_class_name)
            self._d_weights.append(d_class_params["weight"])
            self._d_classification_weights.append(d_class_params["classification_w"])

            curr_d_classified = d_class_params["classified"]
            assert (
                    not curr_d_classified
                    or len(d_class_params["subclasses"]) >= 2
            ), "do you really want to classify into <2 classes?"
            if len(d_class_params["subclasses"]) < 2:
                raise NotImplementedError()  # do we need to store c_ids into target map?

            self._d_classified.append(curr_d_classified)

            # for current d_class_id
            d_c_names = []
            d_c_ws = []
            d_c_tm_ids = []
            for c_id, (c_class_name, c_class_params) in enumerate(d_class_params["subclasses"].items()):
                d_c_names.append(c_class_name)
                d_c_ws.append(c_class_params["weight"])
                for a_name in c_class_params["aliases"]:
                    self._alias2dc[a_name] = (d_id, c_id)
                    self._alias2name[a_name] = c_class_name
                c_tm_id = None
                if curr_d_classified:
                    c_tm_id = n_d_classes + n_c_classes + c_id
                    d_c_tm_ids.append(c_tm_id)
                self._class2tm_id[(d_id, c_id)] = (d_id, c_tm_id)
            self._class2tm_id[(d_id, None)] = (d_id, None)

            self._c_names[d_id] = d_c_names
            self._c_ids[d_id] = np.arange(len(d_c_names))
            if curr_d_classified:
                n_c_classes += len(d_class_params["subclasses"])
                self._c_tm_ids[d_id] = d_c_tm_ids
                self._c_weights[d_id] = d_c_ws

        self._n_d_classes = n_d_classes
        self._n_c_classes = n_c_classes
        self._n_channels = n_d_classes + n_c_classes

        self._d_class_ids = np.arange(len(self._d_names))  # class ids
        self._d_tm_ids = self._d_class_ids  # target map ids

    @property
    def d_class_ids(self):
        return self._d_class_ids

    @property
    def d_tm_ids(self):
        return self._d_tm_ids

    @property
    def d_class_names(self):
        return self._d_names

    @property
    def d_classified(self):
        return self._d_classified

    @property
    def d_weights(self):
        return self._d_weights

    @property
    def d_classification_weights(self):
        return self._d_classification_weights

    @property
    def alias2dc(self):
        return self._alias2dc.copy()

    @property
    def alias2name(self):
        return self._alias2name.copy()

    def get_d_classification_weight(self, d_class_id):
        return self.d_weights[d_class_id]

    def get_c_names(self, d_class_id):
        return self._c_names[d_class_id]

    def get_c_class_ids(self, d_class_id):
        return self._c_ids[d_class_id]

    def get_c_tm_ids(self, d_class_id):
        return self._c_tm_ids[d_class_id]

    def get_c_weights(self, d_class_id):
        return self._c_weights[d_class_id]

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def n_d_channels(self):
        return self._n_d_classes

    @property
    def n_c_channels(self):
        return self._n_c_classes

    def get_tm_ids(self, d_class_id, c_class_id=None):
        return self._class2tm_id[(d_class_id, c_class_id)]
