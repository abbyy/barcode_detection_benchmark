"""
Reads dataset in the specified format and saves it in default format
python data_preparation/make_dataset.py \
    --dataset_path $DATA_DIR \
    --save_data_path $DATASET_DIR \
    --out_dataset $DATASET_INFO_DIR \
    --name $DATASET_NAME \
    --datasets_json $DATASETS_JSON_PATH
(designed to be used by `scripts/prepare_dataset.sh`)
"""

import collections
import json
import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import tqdm.auto as tqdm


class Constants:
    """
    Set of string constants to avoid misprints
    """

    NUM_IMAGES = "num_images"

    class Formats:
        """
        Ground truth file formats keys
        """
        OBJECT = "objects_format"
        MASK = "masks_format"

    class Folders:
        IMAGES = "images_folder"
        OBJECTS = "objects_folder"
        MASKS = "masks_folder"

    class Extensions:
        IMAGE = {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}
        OBJECTS = {".xml"}
        MASKS = {".bmp", ".png", ".jpeg", ".jpg", ".tif", ".tiff"}

    class DatasetColumns:
        """
        Columns names in .csv dataset
        """
        ID = "ID"
        IMAGE = "image"
        OBJECTS = "objects"
        MASK = "mask"

    class DatasetFields:
        """
        Keys in .json general info about dataset
        """
        DATA_PATH = "data_path"
        CSV_PATH = "csv_path"
        INFO_JSON_PATH = "info_json_path"


parser = ArgumentParser()
parser.add_argument(
    '--dataset_path', '-d', type=Path, required=True,
    help="path to dataset root dir"
)
parser.add_argument(
    '--save_data_path', '-s', type=Path, required=True,
    help="path to save dataset copy (if omitted no copy will be saved)"
)
parser.add_argument(
    '--out_dataset_dir', type=Path, required=True,
    help="path to folder where to save .csv parsed dataset"
)
parser.add_argument(
    '--objects_format', '-of', type=str, default='barcode_xml', required=False,
    help="objects markup file format"
)
parser.add_argument(
    '--masks_format', '-mf', type=str, default='barcode_xml', required=False,
    help="masks markup file format"
)
parser.add_argument(
    '--name', '-n', type=str, default='dataset', required=False,
    help="name of the dataset (most likely root folder name)"
)
parser.add_argument(
    '--datasets_json', type=Path, default=None, required=False,
    help="name of the datasets_json (where the data about *all* your datasets is stored"
)
parser.add_argument(
    '--images_folder', type=str, default='Image', required=False,
    help="name of the dataset (most likely root folder name)"
)
parser.add_argument(
    '--objects_folder', type=str, default='Markup', required=False,
    help="name of the dataset (most likely root folder name)"
)
parser.add_argument(
    '--masks_folder', type=str, default=None, required=False,
    help="name of the dataset (most likely root folder name)"
)


def update_json_file(json_filename, update_key, updated_value):
    if os.path.exists(json_filename):
        with open(json_filename) as json_file:
            _json = json.load(json_file)
    else:
        _json = {}
    _json[update_key] = updated_value
    with open(json_filename, "w") as json_file:
        json.dump(_json, json_file, sort_keys=True, indent=4)


def update_datasets_json_file(json_filename, update_key, data_path, csv_path, info_json_path):
    update_json_file(json_filename, update_key, updated_value={
        Constants.DatasetFields.DATA_PATH: data_path,
        Constants.DatasetFields.CSV_PATH: csv_path,
        Constants.DatasetFields.INFO_JSON_PATH: info_json_path
    })


def id_from_fname(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]


def has_extension(filename, allowed_extensions):
    _, ext = os.path.splitext(filename)
    return ext.lower() in allowed_extensions


def has_image_extension(filename):
    return has_extension(filename, allowed_extensions=Constants.Extensions.IMAGE)


def has_objects_extension(filename):
    return has_extension(filename, allowed_extensions=Constants.Extensions.OBJECTS)


def find_in_dir(dirname: str, full_path: bool = False):
    result = [fname for fname in sorted(os.listdir(dirname))]
    if full_path:
        result = [os.path.join(dirname, fname) for fname in result]

    return result


def find_relative_roots(root: str, image_dir_name: str, curr_relative_path: str = ""):
    """returns all relative paths from root with `image_dir_name` subdirectory
    example: (root=root, image_dir_name='Image')
    root/
        dir1/
            dir1a/
                Image/
            dir2a/
                Image/
        dir2/
            Image/
    will return ['dir1/dir1a', 'dir1/dir2a', 'dir2']
    """
    result = []
    for subdir in os.listdir(root):
        if not os.path.isdir(subdir):
            continue
        if subdir == image_dir_name:
            result.append(os.path.join(root, curr_relative_path))
        result.extend(find_relative_roots(root=os.path.join(root, subdir),
                                          image_dir_name=image_dir_name,
                                          curr_relative_path=os.path.join(curr_relative_path, subdir)))
    return result


def find_images_in_dir(dirname: str, full_path: bool = False):
    result = [
        fname
        for fname in find_in_dir(dirname, full_path=full_path)
        if has_image_extension(fname)
    ]
    return result


def find_same_name_allowed_ext(filename, allowed_ext, filenames_set):
    """
    Returns
        None if object is not found
        filename if single filename with such conditions found
        raises AssertionError if several filenames satisfies the condition
    :param filename:
    :param allowed_ext:
    :param filenames_set:
    :return:
    """
    filename_without_ext = os.path.splitext(filename)[0]
    filename_candidates = {f"{filename_without_ext}{ext}" for ext in allowed_ext}
    filename_candidates = \
        filename_candidates.union({f"{filename_without_ext}{ext.upper()}" for ext in allowed_ext})
    filename_candidates = filename_candidates.intersection(filenames_set)
    assert len(filename_candidates) < 2

    if filename_candidates:
        return list(filename_candidates)[0]
    return None


def get_dataset_info(args):
    info = {
        # directories
        Constants.Folders.IMAGES: args.images_folder,
        Constants.Folders.OBJECTS: args.objects_folder,
        Constants.Folders.MASKS: args.masks_folder,
        # file formats
        Constants.Formats.MASK: args.masks_format,
        Constants.Formats.OBJECT: args.objects_format,
    }

    return info


def main(args):
    os.makedirs(args.save_data_path, exist_ok=True)
    os.makedirs(args.out_dataset_dir, exist_ok=True)

    dataset_info = get_dataset_info(args)

    # for image_file, markup_file add them into dataset
    samples = collections.defaultdict(dict)
    # for key in ("images", "objects", "masks"):
    images_folder = dataset_info[Constants.Folders.IMAGES]
    objects_folder = dataset_info[Constants.Folders.OBJECTS]
    masks_folder = dataset_info[Constants.Folders.MASKS]

    objects_files = set(
        os.listdir(args.dataset_path / objects_folder)
    ) if objects_folder is not None else set()
    masks_files = set(
        os.listdir(args.dataset_path / masks_folder)
    ) if masks_folder is not None else set()
    assert objects_files or masks_files, "no markup data"

    for fname in tqdm.tqdm(find_images_in_dir(args.dataset_path / images_folder)):
        full_fname = os.path.join(images_folder, fname)
        sample_id = id_from_fname(full_fname)

        sample = {
            Constants.DatasetColumns.ID: sample_id,
            Constants.DatasetColumns.IMAGE: full_fname,
        }
        if objects_folder is not None:
            found_filename = find_same_name_allowed_ext(
                filename=fname,
                allowed_ext=Constants.Extensions.OBJECTS,
                filenames_set=objects_files
            )
            sample[Constants.DatasetColumns.OBJECTS] = os.path.join(objects_folder, found_filename)
        if masks_folder is not None:
            found_filename = find_same_name_allowed_ext(
                filename=fname,
                allowed_ext=Constants.Extensions.MASKS,
                filenames_set=masks_files
            )
            sample[Constants.DatasetColumns.MASK] = os.path.join(objects_folder, found_filename)

        samples[sample_id].update(sample)

    dataframe = pd.DataFrame.from_dict(samples, orient="index")

    isna_row = dataframe.isna().any(axis=1)
    if isna_row.any():
        # we have to know 1)image filename 2)object/mask filename - if there is no match at some row
        # it is likely that some of this happened:
        # 1)the corresponding image/object/mask filename is missing in the specified directory or
        # 2)we had some errors in the above processing
        fname = args.out_dataset_dir / "nans.json"
        dataframe[isna_row].to_json(fname, orient="records")
        dataframe.dropna(axis=0, how="any", inplace=True)
        if isna_row.any():
            logging.warning(f"There were some NaNs in the resulted dataset.csv, check {fname}")

    dataset_info[Constants.NUM_IMAGES] = dataframe.shape[0]
    csv_dataset_path = args.out_dataset_dir / "dataset.csv"
    dataframe.to_csv(csv_dataset_path, index=False)

    info_json_path = args.out_dataset_dir / "dataset_info.json"
    with open(str(info_json_path), 'w') as f:
        json.dump(dataset_info, f, sort_keys=True, indent=4)
    with open(str(args.out_dataset_dir / "dataset_info.pkl"), 'wb') as f:
        pickle.dump(dataset_info, f)

    if args.datasets_json is not None:
        # add this dataset to all processed datasets
        update_datasets_json_file(str(args.datasets_json), update_key=args.name,
                                  data_path=str(args.dataset_path),
                                  csv_path=str(csv_dataset_path),
                                  info_json_path=str(info_json_path))


def _main():
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    _main()
