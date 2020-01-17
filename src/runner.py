# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Custom catalyst runners
"""
from catalyst.dl import SupervisedRunner
from catalyst.utils.typing import Model, Device


class CustomRunner(SupervisedRunner):
    def __init__(self, model: Model = None, device: Device = None, input_key: str = "image",
                 output_key: str = "logits", input_target_key: str = "targets"):
        super().__init__(model, device, input_key, output_key, input_target_key)
