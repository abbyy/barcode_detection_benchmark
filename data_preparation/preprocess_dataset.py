"""
Resize all images and bboxes to lower resolution to speed up training
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import xml.etree.ElementTree as ET

import cv2
import tqdm.auto as tqdm

from make_dataset import find_images_in_dir, find_same_name_allowed_ext

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))
from data.transforms import LongestMaxSizeMinMultiple

parser = ArgumentParser()
parser.add_argument(
    '--in_path', type=Path, required=True,
    help="path to class config"
)
parser.add_argument(
    '--out_path', type=Path, required=True,
    help="path to save dataset copy (if omitted no copy will be saved)"
)
parser.add_argument(
    '--images_dir', type=str, default="Image",
    help="path to folder where to save all configs"
)
parser.add_argument(
    '--objects_dir', type=str, default="Markup",
    help="path to folder where to save all configs"
)
parser.add_argument(
    '--max_side', type=int, default=1024,
    help="max side of resized image"
)


def main(
        max_side,
        in_path,
        out_path,
        images_dir='Image',
        objects_dir='Markup',
):
    # os.makedirs(out_path)
    in_images_dir = in_path / images_dir
    in_objects_dir = in_path / objects_dir

    out_images_dir = out_path / images_dir
    out_objects_dir = out_path / objects_dir
    os.makedirs(out_objects_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    t = LongestMaxSizeMinMultiple(max_size=max_side, min_multiple=1)

    markup_fnames = set(os.listdir(in_objects_dir))
    for image_name in tqdm.tqdm(find_images_in_dir(in_images_dir)):
        image = cv2.imread(os.path.join(in_images_dir, image_name))
        markup_fname = find_same_name_allowed_ext(image_name, allowed_ext=['.xml'], filenames_set=markup_fnames)
        objects_xml = ET.parse(str(in_objects_dir / markup_fname))
        transformed = t(image=image)
        resized_image = transformed["image"]
        x_scale, y_scale = transformed["x_scale"], transformed["y_scale"]
        for point in objects_xml.getiterator("Point"):
            point.attrib["X"] = str(x_scale * float(point.attrib["X"]))
            point.attrib["Y"] = str(y_scale * float(point.attrib["Y"]))
        cv2.imwrite(str(out_images_dir / image_name), resized_image)
        objects_xml.write(str(out_objects_dir / markup_fname))


def _main():
    args = parser.parse_args()
    main(
        max_side=args.max_side,
        in_path=args.in_path,
        out_path=args.out_path,
        images_dir=args.images_dir,
        objects_dir=args.objects_dir
    )


if __name__ == '__main__':
    _main()
