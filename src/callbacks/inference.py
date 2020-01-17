# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
Callbacks for inference predictions
"""
import json
import os

from catalyst.dl import RunnerState
from catalyst.dl.core import LoggerCallback


class InferenceCallback(LoggerCallback):
    """
    Saving results on inference
    """

    def __init__(self, order: int = None,
                 results_dir: str = "results",
                 relative: bool = True,
                 outpath_key: str = "image_name",
                 input_objects_key: str = "objects",
                 output_objects_key: str = "objects",
                 metrics_filename: str = "metrics.json"):
        super().__init__(order)
        self.results_dir = results_dir
        self.relative = relative
        self.metrics_filename = metrics_filename

        self.outpath_key = outpath_key
        self.input_objects_key = input_objects_key
        self.output_objects_key = output_objects_key

    def on_loader_end(self, state: RunnerState):
        base_dir = os.path.join(self.results_dir, state.loader_name)
        if self.relative:
            base_dir = os.path.join(state.logdir, base_dir)
        os.makedirs(base_dir, exist_ok=True)
        metrics_filename = os.path.join(base_dir, self.metrics_filename)

        with open(metrics_filename, "w") as f:
            json.dump(state.metrics.epoch_values[state.loader_name], f, sort_keys=True, indent=4)

    def on_batch_end(self, state: RunnerState):
        image_names = state.input[self.outpath_key]
        original_objects = state.input[self.input_objects_key]
        detected_objects = state.output[self.output_objects_key]

        for image_name, gt_objects, found_objects in zip(image_names, original_objects, detected_objects):
            for folder, objects in zip(("original", "detected"), (gt_objects, found_objects)):
                self._save_to_file(objects, self._get_filename(state, folder, image_name))

    def _get_filename(self, state: RunnerState, folder: str, image_name: str):
        if self.relative:
            base_dir = os.path.join(state.logdir, self.results_dir)
        else:
            base_dir = self.results_dir
        base_dir = os.path.join(base_dir, state.loader_name, folder)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, image_name + ".txt")

    @staticmethod
    def _save_to_file(objects, filename):
        with open(filename, "w") as f:
            f.writelines(map(
                lambda o: ", ".join(map(str, o.location.flatten())) + f", {o.class_name}\n",
                objects
            ))
