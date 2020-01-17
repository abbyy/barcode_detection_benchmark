# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Raw network output postprocessing
"""
import torch
import torch.nn.functional as F
from catalyst.dl import Callback, CallbackOrder, RunnerState

import utils
from converter import Converter
from data.data_info import TargetMapInfo


class MaybeUpscaleOutputCallback(Callback):
    """
    Upscale network predictions

    Useful if your network predicts output which has size less than target (e.g. you have poolings or strided convs)
    Makes your training more stable [1] and your postprocessing results more precise
    But training & inference & postprocessing become slightly slower due to larger output tensor
    """

    def __init__(
            self,
            input_tensor_key: str = "mask",
            output_tensor_key: str = "logits",
            inplace: bool = True,
            old_output_key: str = None,
            new_output_key: str = None,
            mode: str = "bilinear",
            align_corners: bool = False
    ):
        """

        :param input_tensor_key: key for tensor in input used to resize output to same (H, W)
        :param output_tensor_key: key for output tensor to be resized
        :param inplace: save resized tensor on same key
        :param old_output_key: if not None - save tensor before reshape by specified key
        :param new_output_key: if not None (and not inplace) - save reshaped tensor by this key
            in the `inplace` mode new_output_key will be set to output_tensor_key
        :param mode:
        """
        super().__init__(order=CallbackOrder.Internal)
        self.input_tensor_key = input_tensor_key
        self.output_tensor_key = output_tensor_key
        self.old_output_key = old_output_key if inplace else None
        self.new_output_key = output_tensor_key if inplace else new_output_key
        assert self.new_output_key is not None
        assert mode in ["bilinear", "nearest"]
        self.mode = mode
        self.align_corners = align_corners

    def on_batch_end(self, state: RunnerState):
        target_size = state.input[self.input_tensor_key].size()[2:]  # (B, C, H, W) -> (H, W)
        output = state.output[self.output_tensor_key]
        if self.old_output_key:
            state.output[self.old_output_key] = output
        if output.size()[2:] != target_size:
            output = F.interpolate(output, size=target_size, mode=self.mode, align_corners=self.align_corners)
        state.output[self.new_output_key] = output


class RawMaskPostprocessingCallback(Callback):
    """
    Network output logits -> heatmaps & detected objects
    """

    def __init__(
            self,
            threshold: float = 0.5,
            input_key: str = "logits",
            output_mask_key: str = "heatmaps",
            output_objects_key: str = "objects",
            input_x_scale_key: str = "x_scale",
            input_y_scale_key: str = "y_scale",
            output_objects_on_original_key: str = None
    ):
        super().__init__(CallbackOrder.Internal)
        self.threshold = threshold
        self.input_key = input_key
        self.output_mask_key = output_mask_key
        self.output_objects_key = output_objects_key

        self.input_x_scale_key = input_x_scale_key
        self.input_y_scale_key = input_y_scale_key
        self.output_objects_on_original_key = output_objects_on_original_key

        self.converter = Converter(TargetMapInfo())

    def on_batch_end(self, state: RunnerState):
        batch_sigmoid_mask = torch.sigmoid(state.output[self.input_key]).detach().cpu().numpy()

        detected_objects = self.converter.postprocess_target_map(batch_sigmoid_mask)
        state.output[self.output_objects_key] = detected_objects
        state.output[self.output_mask_key] = batch_sigmoid_mask

        if self.output_objects_on_original_key is not None:
            x_scale = 1.0 / state.input[self.input_x_scale_key]
            y_scale = 1.0 / state.input[self.input_y_scale_key]
            if isinstance(x_scale, torch.Tensor):
                x_scale = x_scale.cpu().numpy()
                y_scale = y_scale.cpu().numpy()
            scaled_objects = [
                utils.rescale_objects(objects, xs, ys)
                for objects, xs, ys in zip(detected_objects, x_scale, y_scale)
            ]
            state.output[self.output_objects_on_original_key] = scaled_objects
