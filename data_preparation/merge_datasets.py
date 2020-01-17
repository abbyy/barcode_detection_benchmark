"""
Merge registered (by make_dataset.py or update_datasets_json.py) datasets by common prefix into single dataset
"""
import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from make_dataset import Constants, update_datasets_json_file

parser = argparse.ArgumentParser()
parser.add_argument(
    '--aggregated_name', '-n', type=str, default='synth', required=True,
    help="key prefix in datasets_json to be aggregated by train/valid splits"
)
parser.add_argument(
    '--datasets_json', type=Path, default=None, required=True,
    help="name of the datasets_json (where the data about *all* your datasets is stored"
)
parser.add_argument(
    '--out_dataset_dir', type=Path, required=True,
    help="path to folder where to save .csv parsed dataset"
)


def read_info_jsons(all_datasets_json, keys):
    jsons = []
    for key in keys:
        json_path = all_datasets_json[key][Constants.DatasetFields.INFO_JSON_PATH]
        with open(json_path, "r") as f:
            jsons.append(json.load(f))
    return jsons


def merge_info_jsons(jsons):
    assert len(jsons) > 0
    first_json = jsons[0]
    for j in jsons[1:]:
        assert len(first_json) == len(j)
        assert first_json[Constants.Formats.OBJECT] == j[Constants.Formats.OBJECT]
        assert first_json[Constants.Formats.MASK] == j[Constants.Formats.MASK]
    agg_json = copy.deepcopy(first_json)
    agg_json[Constants.NUM_IMAGES] = sum(j[Constants.NUM_IMAGES] for j in jsons)
    return agg_json


def main(args):
    os.makedirs(args.out_dataset_dir, exist_ok=True)

    assert os.path.exists(args.datasets_json)
    with open(args.datasets_json, "r") as f:
        all_datasets_infos = json.load(f)

    dataset_name = args.aggregated_name

    related_datasets = list(filter(lambda k: k.startswith(dataset_name), all_datasets_infos.keys()))

    group_full_name = "full"
    groups = ["train", "valid", "infer", group_full_name]

    # find aggregated keys for each group
    group2info_keys = defaultdict(list)
    for d in related_datasets:
        for g in groups:
            if g == group_full_name and d != dataset_name:
                group2info_keys[g].append(d)
            elif d.endswith(g):
                if f"{dataset_name}_{g}" != d:  # this value will be overriden if present
                    group2info_keys[g].append(d)
                break  # "full" is the last one
    for g in groups[:2]:
        assert len(group2info_keys[g]) == len(group2info_keys[group_full_name])

    for group_name, group_info_keys in group2info_keys.items():
        # save formats info
        group_info_jsons = read_info_jsons(all_datasets_infos, keys=group_info_keys)
        group_info_json = merge_info_jsons(group_info_jsons)
        group_json_path = args.out_dataset_dir / f"{group_name}.json"
        with open(group_json_path, "w") as f:
            json.dump(group_info_json, f, sort_keys=True, indent=4)
        # compute NEW data path
        data_root_path = os.path.commonpath(
            [all_datasets_infos[k][Constants.DatasetFields.DATA_PATH] for k in group_info_keys])
        # merge dataframes
        df = pd.DataFrame()
        for k in group_info_keys:
            in_dataset = pd.read_csv(all_datasets_infos[k][Constants.DatasetFields.CSV_PATH], sep=',')
            relative_path = os.path.relpath(
                all_datasets_infos[k][Constants.DatasetFields.DATA_PATH],
                start=data_root_path
            )
            key_id = "s" + k[len(dataset_name):]  # "synth01" -> "s01"
            in_dataset[Constants.DatasetColumns.ID] = \
                in_dataset[Constants.DatasetColumns.ID].apply(lambda _id: f"{key_id}_{_id}")
            for column in in_dataset.columns:
                if column not in [Constants.DatasetColumns.ID, "fold"]:
                    in_dataset[column] = in_dataset[column].apply(
                        lambda value: os.path.normpath(os.path.join(relative_path, value)))
            df = df.append(in_dataset)
        # merged
        group_csv_path = args.out_dataset_dir / f"{group_name}.csv"
        df.to_csv(group_csv_path, index=False)
        # json saved, csv saved, root data computed
        # now write info to all datasets json
        save_key = dataset_name if group_name == "full" else f"{dataset_name}_{group_name}"
        update_datasets_json_file(
            json_filename=args.datasets_json,
            update_key=save_key,
            data_path=str(data_root_path),
            csv_path=str(group_csv_path),
            info_json_path=str(group_json_path)
        )


def _main():
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    _main()
