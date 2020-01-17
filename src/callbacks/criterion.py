# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Custom criterion callbacks
"""
from typing import Union, List, Any

import torch
from catalyst.dl import RunnerState
from catalyst.dl.callbacks import CriterionCallback, CriterionAggregatorCallback


class PretransformCriterionCallback(CriterionCallback):
    """
    loss_value = criterion(input_transform(input), output_transform(output))
    """

    def _compute_loss(self, state: RunnerState, criterion):
        output = self._get(state.output, self.output_key)
        input = self._get(state.input, self.input_key)

        output = self.transform_output(output)
        input = self.transform_input(input)

        loss = criterion(output, input)
        return loss

    def transform_input(self, input):
        return input

    def transform_output(self, output):
        return output


class SelectedChannelCriterionCallback(PretransformCriterionCallback):
    """
    Criterion applied for the selected output channel only
    """

    def __init__(self, input_key: Union[str, List[str]] = "targets", output_key: Union[str, List[str]] = "logits",
                 prefix: str = "loss", criterion_key: str = None, multiplier: float = 1.0,
                 target_map_id: int = None, input_channel_id: int = None, output_channel_id: int = None):
        super().__init__(input_key, output_key, prefix, criterion_key, multiplier)
        if target_map_id is not None:
            self.input_selected_channel = target_map_id
            self.output_selected_channel = target_map_id
        else:
            assert input_channel_id is not None or output_channel_id is not None
            self.input_selected_channel = input_channel_id
            self.output_selected_channel = output_channel_id

    def transform_input(self, input):
        if self.input_selected_channel is not None:
            return input[:, self.input_selected_channel]
        return input

    def transform_output(self, output):
        if self.output_selected_channel is not None:
            return output[:, self.output_selected_channel]
        return output


class WeightedCriterionAggregatorCallback(CriterionAggregatorCallback):
    """
    Weighted criterion aggregation
    """

    def __init__(self, prefix: str, loss_keys: Union[str, List[str]] = None,
                 loss_aggregate_fn: str = "mean",
                 weights: List[float] = None,
                 multiplier: float = 1.0) -> None:
        super().__init__(prefix, loss_keys, loss_aggregate_fn="sum", multiplier=multiplier)
        # note that we passed `loss_aggregate_fn="sum"` always to reuse parent's `on_batch_end` method unchanged
        # we use "sum" as after `_preprocess_loss` individual losses are already weighted and we need to sum them

        assert self.loss_keys is not None
        assert weights is not None and len(weights) == len(self.loss_keys)
        self.weights = weights
        if loss_aggregate_fn == "mean":
            self.weights = [w / sum(weights) for w in weights]

    def _preprocess_loss(self, loss: Any) -> List[torch.Tensor]:
        assert isinstance(loss, dict)
        return [loss[key] * self.weights[i] for i, key in enumerate(self.loss_keys)]
