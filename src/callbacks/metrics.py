# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Metric calculation callbacks
"""
from typing import Callable, List

from catalyst.dl import RunnerState
from catalyst.dl.callbacks import ConfusionMatrixCallback
from catalyst.dl.core import MetricCallback

from evaluation import DatasetMetricCalculator


class SegmentationConfusionMatrixCallback(ConfusionMatrixCallback):
    """
    Confusion matrix tensorboard for segmentation
    """

    def _add_to_stats(self, outputs, targets):
        assert outputs.size() == targets.size(), "non-equal shapes of outputs and targets"
        if outputs.size(1) == 1:
            outputs = (outputs.reshape((-1,)) > 0).long()
            targets = targets.reshape((-1,))
        else:
            outputs = outputs.argmax(dim=1).reshape((-1,))
            targets = targets.argmax(dim=1).reshape((-1,))
        super()._add_to_stats(outputs, targets)


class ObjectDetectionMetricsCallback(MetricCallback):
    """
    Various object detection metrics (computed for entire loader once in epoch)
    - precision/recall/f1/detection_rate at different iou_thresholds
    - classification accuracy of detected objects
    """

    def __init__(
            self,
            prefix: str = "",
            metric_fn: Callable = None,
            input_key: str = "objects",
            input_target_key: str = "mask",
            output_logits_key: str = "logits",
            output_key: str = "objects",
            filter_only: int = None,  # Barcode d_class_id
            output_metric_key: str = "image_metrics",  # "image_metrics_Barcode"
            class_names: List[str] = None,  # Barcode class_names
            class_tm_indices: List[int] = None,  # Barcode class_tm_ids
            compute_classification_metrics: bool = False,  # Barcode d_classified
            **metric_params
    ):
        super().__init__(prefix, metric_fn, input_key, output_key, **metric_params)
        self.input_target_key = input_target_key
        self.output_logits_key = output_logits_key
        self.output_metric_key = output_metric_key

        self.filter_only = filter_only

        self.calculator = None
        self.class_names = class_names
        self.compute_classification_metrics = compute_classification_metrics
        self.class_tm_indices = class_tm_indices

    def on_loader_start(self, state: RunnerState):
        self.calculator = DatasetMetricCalculator(
            class_names=self.class_names,
            compute_classification_metrics=self.compute_classification_metrics)

    def on_loader_end(self, state: RunnerState):
        scalar_logs = self.calculator.get_metrics()
        for name, value in scalar_logs.items():
            # state.metrics.add_batch_value(name=f"{self.prefix}{name}", value=value)
            state.metrics.epoch_values[state.loader_name][f"{self.prefix}{name}"] = value

    def on_batch_end(self, state: RunnerState):
        gt_objects = state.input[self.input_key]
        found_objects = state.output[self.output_key]
        if self.filter_only is not None:
            gt_objects = [list(filter(
                lambda o: o.d_class_id == self.filter_only,
                img_objects
            )) for img_objects in gt_objects]
            found_objects = [list(filter(
                lambda o: o.d_class_id == self.filter_only,
                img_objects
            )) for img_objects in found_objects]

        gt_segmap = None
        pr_segmap = None
        if self.compute_classification_metrics:
            gt_segmap = state.input[self.input_target_key][:, self.class_tm_indices].detach().cpu().numpy()
            pr_segmap = state.output[self.output_logits_key][:, self.class_tm_indices].detach().cpu().numpy()
        image_metrics, _ = self.calculator.evaluate_batch(gt_objects, found_objects,
                                                          gt_segmap=gt_segmap, classification_logits=pr_segmap)

        state.output[self.output_metric_key] = image_metrics
