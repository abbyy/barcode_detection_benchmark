# Copyright (С) ABBYY Production LLC, 2019. All rights reserved.
"""
All kinds of visualizations
"""
import os
import shutil
from pathlib import Path
from typing import Union, List
from collections import Counter

import imageio
import matplotlib.pyplot as plt
import numpy as np
from catalyst.dl import Callback, CallbackOrder, RunnerState
from catalyst.utils.image import tensor_to_ndimage

from converter import Converter
from data.data_info import TargetMapInfo
from evaluation import ImageResultCategories


class Visualizer:
    """
    Class for drawing heatmaps & masks
    """

    # RGB colors to draw
    colors = [plt.cm.tab20(i, bytes=True)[:3] for i in range(21)]

    @staticmethod
    def get_color(class_id):
        return Visualizer.colors[class_id % len(Visualizer.colors)]

    @staticmethod
    def draw_hitmap(heatmap, image, min_value=0.0, max_value=1.0, h_strength=0.5):
        assert heatmap.ndim == 2
        rgb_heatmap = np.zeros(heatmap.shape[:2] + (3,), dtype=np.uint8)
        rgb_heatmap[:, :, 0] = ((heatmap - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        rgb_heatmap[:, :, 2] = 255 - rgb_heatmap[:, :, 0]

        assert image.ndim == 3 and image.shape[-1] == 3
        assert image.dtype == np.uint8
        assert rgb_heatmap.shape == image.shape
        heatmap_on_image = (
                h_strength * rgb_heatmap.astype(np.float) + (1 - h_strength) * image.astype(np.float)
        ).astype(np.uint8)
        return heatmap_on_image

    @staticmethod
    def draw_target_on_image(target, image, mask_strength: float = 0.8):
        assert image.dtype == np.uint8 and target.dtype == np.uint8
        assert target.shape[:2] == image.shape[:2]
        assert target.ndim == 2
        image = image.astype(np.float)

        for cls_id in np.unique(target):
            if cls_id > 0:
                draw_color = np.array(Visualizer.get_color(cls_id))
                mask = target == cls_id
                image[mask] *= (1 - mask_strength)
                image[mask] += mask_strength * draw_color
        return image.astype(np.uint8)


class OriginalImageSaverCallback(Callback):
    """
    Callback to save input images into folder
    """

    def __init__(
            self,
            output_dir: str,
            relative: bool = True,
            filename_suffix: str = "",
            filename_extension: str = ".jpg",
            input_key: str = "image",
            outpath_key: str = "name",
    ):
        super().__init__(CallbackOrder.Other)
        self.output_dir = Path(output_dir)
        self.relative = relative
        self.filename_suffix = filename_suffix
        self.filename_extension = filename_extension
        self.input_key = input_key
        self.outpath_key = outpath_key

    def get_image_path(self, state: RunnerState, name: str, suffix: str = ""):
        if self.relative:
            out_dir = Path(state.logdir) / self.output_dir
        else:
            out_dir = self.output_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        res = out_dir / f"{name}{suffix}{self.filename_extension}"

        return res

    def on_batch_end(self, state: RunnerState):
        names = state.input[self.outpath_key]
        images = state.input[self.input_key].cpu()
        images = tensor_to_ndimage(images, dtype=np.uint8)

        for image, name in zip(images, names):
            fname = self.get_image_path(state, name, self.filename_suffix)
            imageio.imwrite(fname, image)


class OverlayMaskImageSaverCallback(OriginalImageSaverCallback):
    """
    Callback to save mask on image into folder
    """

    def __init__(
            self,
            output_dir: str,
            relative: bool = True,
            mask_strength: float = 0.5,
            filename_suffix: str = "",
            filename_extension: str = ".jpg",
            input_key: str = "image",
            mask_key: str = "mask",
            mask_in_input: bool = False,
            outpath_key: str = "name"
    ):
        super().__init__(
            output_dir=output_dir,
            relative=relative,
            filename_suffix=filename_suffix,
            filename_extension=filename_extension,
            input_key=input_key,
            outpath_key=outpath_key,
        )
        self.mask_strength = mask_strength
        self.mask_key = mask_key
        self.mask_in_input = mask_in_input

    def on_batch_end(self, state: RunnerState):
        names = state.input[self.outpath_key]
        images = tensor_to_ndimage(state.input[self.input_key].cpu())
        if self.mask_in_input:
            masks = state.input[self.mask_key].cpu().squeeze(1).numpy()
        else:
            masks = state.output[self.mask_key]

        for name, image, mask in zip(names, images, masks):
            image = Visualizer.draw_target_on_image(mask, image, self.mask_strength)
            fname = self.get_image_path(state, name, self.filename_suffix)
            imageio.imwrite(fname, image)


class VisualizationsSaverCallback(Callback):
    """
    Callback to save *pretty much everything* into folder
    Draw & save:
    - original image (.orig)
    - input masks (.m_in) [x n_channels]
    - output masks (.m_out) [x n_channels]
    - input objects (.o_in) [x n_channels]
    - output objects (.o_out) [x n_channels]

    Information is stored only if image satisfies some criteria, e.g. recall on that image < 1,
        these criteria may be set via "kinds" parameter
    """

    def __init__(
            self,
            kinds: Union[str, List[str]] = "all",
            iou_threshold: float = 0.5,
            input_image_key: str = "image",
            input_mask_key: str = "mask",
            input_image_name_key: str = "image_name",
            output_mask_key: str = "mask",
            output_image_metrics_key: str = "image_metrics",
            output_objects_key: str = "objects",
            output_dir: str = "visualizations",
            filename_extension: str = ".jpg",
            mask_strength: float = 0.5,
            detection_only: str = True,
            valid_only: str = True,
            max_images: int = None
    ):
        """

        :param kinds: list of criteria for image to be saved
            all visualizations will be saved for the corresponding folders of each *kind* which this image satisfies
        :param iou_threshold: threshold value for image metrics to estimate precision/recall/etc
            used while checking if image satisfies some criteria or not
        :param input_image_key:
        :param input_mask_key:
        :param input_image_name_key:
        :param output_mask_key:
        :param output_image_metrics_key:
        :param output_objects_key:
        :param output_dir:
        :param filename_extension:
        :param mask_strength: overlay between mask and original image
            approximately visualization are computed as `mask * strength + image * (1-strength)`
        :param detection_only: draw only masks for detected classes (not classification classes)
            may be useful to save disk space
        :param valid_only: draw only on validation stage
        :param max_images: draw visualisations for at most max_images per loader
        """
        super().__init__(CallbackOrder.Other)
        self.kinds = ImageResultCategories(kinds)
        self.iou_threshold = iou_threshold
        self.input_image_key = input_image_key
        self.input_mask_key = input_mask_key
        self.input_image_name_key = input_image_name_key
        self.output_mask_key = output_mask_key
        self.output_image_metrics_key = output_image_metrics_key
        self.output_objects_key = output_objects_key
        self.output_dir = output_dir
        self.filename_extension = filename_extension
        self.mask_strength = mask_strength

        self.target_map_info = TargetMapInfo()
        self.converter = Converter(self.target_map_info)

        self.detection_only = detection_only
        self.valid_only = valid_only
        self.max_images = max_images

        self._categories_counter = None

    def on_loader_start(self, state: RunnerState):
        # create visualizations directories or clear them if already exists
        out_dir = os.path.join(state.logdir, self.output_dir, state.loader_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)  # True will make you able to look into folders while training

        for out_subdir in self.kinds.get_folders():
            curr_folder = os.path.join(out_dir, out_subdir)
            os.makedirs(curr_folder, exist_ok=True)
        self._categories_counter = Counter()

    def get_image_path(self, state, curr_dir, filename, filename_suffix):
        return os.path.join(state.logdir, self.output_dir, state.loader_name, curr_dir,
                            f"{filename}.{filename_suffix}{self.filename_extension}")

    def on_batch_end(self, state: RunnerState):
        if self.valid_only and state.need_backward:
            return

        names = state.input[self.input_image_name_key]
        images = tensor_to_ndimage(state.input[self.input_image_key].cpu(), dtype=np.uint8)
        in_masks = state.input[self.input_mask_key].cpu().numpy()
        out_masks = state.output[self.output_mask_key]
        in_objects = state.input[self.output_objects_key]
        out_objects = state.output[self.output_objects_key]

        image_metrics = state.output[self.output_image_metrics_key]
        for image_idx, image in enumerate(images):
            name = names[image_idx]
            metrics = image_metrics[image_idx]

            assert self.iou_threshold in metrics
            vis_folders = self._get_categories(metrics[self.iou_threshold])

            if not vis_folders:
                continue

            # original
            self.save_everywhere(state, vis_folders, name, "img", image)
            # masks
            for suffix, mask in zip(("m_in", "m_out"), (in_masks[image_idx], out_masks[image_idx])):
                for d_cls_id, d_tm_id, d_name, d_classified in zip(
                        self.target_map_info.d_class_ids,
                        self.target_map_info.d_tm_ids,
                        self.target_map_info.d_class_names,
                        self.target_map_info.d_classified
                ):
                    save_img = Visualizer.draw_hitmap(mask[d_tm_id], image)
                    self.save_everywhere(state, vis_folders, name, f"{d_name}.{suffix}", save_img)
                    if d_classified and not self.detection_only:
                        for c_name, c_tm_id in zip(
                                self.target_map_info.get_c_names(d_cls_id),
                                self.target_map_info.get_c_tm_ids(d_cls_id)
                        ):
                            save_img = Visualizer.draw_hitmap(mask[c_tm_id], image)
                            self.save_everywhere(state, vis_folders, name, f"{c_name}.{suffix}", save_img)
            # objects
            for suffix, objects in zip(("o_in", "o_out"), (in_objects[image_idx], out_objects[image_idx])):
                objects_mask = self.converter.build_target_map(objects,
                                                               image_size=image.shape[:2],
                                                               for_visualization=True)
                save_img = Visualizer.draw_target_on_image(objects_mask, image)
                self.save_everywhere(state, vis_folders, name, suffix, save_img)

    def save_everywhere(self, state, folders, name, suffix, save_image):
        for folder in folders:
            fname = self.get_image_path(state, folder, name, suffix)
            imageio.imwrite(fname, save_image)

    def _get_categories(self, metrics):
        vis_folders = self.kinds.get_categories(metrics)
        if self.max_images is not None:
            vis_folders = list(filter(lambda c: self._categories_counter[c] < self.max_images, vis_folders))
            self._categories_counter.update(vis_folders)
        return vis_folders

