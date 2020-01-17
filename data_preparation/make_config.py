"""
Build YAML experiment config from template
python data_preparation/make_config.py \
    --class_config $CONFIG_CLASS \
    --in_template $CONFIG_IN_TEMPLATE \
    --out_config_dir $CONFIG_DIR \
    --all_datasets_json $DATASETS_JSON_PATH \
    --train=$TRAIN \
    --valid=$VALID \
    --infer $INFER \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --image_size $IMAGE_SIZE \
    --max_caching_size $MAX_CACHING_SIZE
(designed to be used by `scripts/prepare_config.sh`)
"""
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from catalyst.utils import load_ordered_yaml
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))
from data.data_info import TargetMapInfo
from utils import process_class_config
from make_dataset import Constants

parser = ArgumentParser()
parser.add_argument(
    '--class_config', type=Path, required=True,
    help="path to class config"
)
parser.add_argument(
    '--in_template', type=Path, required=True,
    help="path to save dataset copy (if omitted no copy will be saved)"
)
parser.add_argument(
    '--out_config_dir', type=Path, required=True,
    help="path to folder where to save all configs"
)
parser.add_argument(
    '--out_config_name', type=str, default="config.yml",
    help="name of generated output config"
)
parser.add_argument(
    '--all_datasets_json', type=str, required=True,
    help="path to json with all prepared datasets information"
)
parser.add_argument(
    '--train', type=str, required=True,
    help="names of training datasets keys in all_datasets_json, separated by space"
)
parser.add_argument(
    '--valid', type=str, required=True,
    help="names of training datasets keys in all_datasets_json, separated by space"
)
parser.add_argument(
    '--infer', type=str, required=False, nargs='*',
    help="names of training datasets keys in all_datasets_json, separated by space"
)
parser.add_argument(
    '--batch_size', type=int, default=8
)
parser.add_argument(
    '--num_epochs', type=int, default=500,
)
parser.add_argument(
    '--num_workers', type=int, default=4
)
parser.add_argument(
    '--image_size', type=int, default=512
)
parser.add_argument(
    '--max_caching_size', type=int, default=1000
)
parser.add_argument(
    '--dilated_model', type=bool, default=False
)


class DatasetInfo:
    def __init__(self, json_responce):
        self.datapath = json_responce[Constants.DatasetFields.DATA_PATH]
        self.csv_path = json_responce[Constants.DatasetFields.CSV_PATH]
        self.info_json_path = json_responce[Constants.DatasetFields.INFO_JSON_PATH]


def main(
        in_template,
        class_config_path,
        out_config_dir,
        out_config_name,
        batch_size,
        num_epochs,
        num_workers,
        image_size,
        max_caching_size,
        all_datasets_json,
        train, valid, infer=None,
        dilated_model=False
):
    assert os.path.exists(in_template)
    assert os.path.exists(class_config_path)

    os.makedirs(out_config_dir, exist_ok=True)
    shutil.copy(in_template, out_config_dir / "_template.yml")
    shutil.copy(class_config_path, out_config_dir / "class_config.yml")

    # read class config
    with open(class_config_path, "r") as fin:
        class_config = load_ordered_yaml(fin)
    # set default values
    class_config = process_class_config(class_config["class_config"])
    target_map_info = TargetMapInfo(class_config)

    # all datasets json
    with open(all_datasets_json, "r") as json_file:
        _json = json.load(json_file)
    train_dataset = DatasetInfo(_json[train]) if train else None
    valid_dataset = DatasetInfo(_json[valid]) if valid else None
    infer_datasets = {name: DatasetInfo(_json[name]) for name in infer}

    # processing template
    env = Environment(
        loader=FileSystemLoader(str(in_template.absolute().parent)),
        trim_blocks=True,
        lstrip_blocks=True
    )
    env.globals.update(zip=zip)  # enable zip command inside jinja2 template
    template = env.get_template(in_template.name)

    out_config = out_config_dir / out_config_name
    out_config.write_text(
        template.render(
            tm=target_map_info,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            infer_datasets=infer_datasets,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            max_caching_size=max_caching_size,
            dilated_model=dilated_model
        )
    )


def _main():
    args = parser.parse_args()
    main(
        in_template=args.in_template,
        class_config_path=args.class_config,
        out_config_dir=args.out_config_dir,
        out_config_name=args.out_config_name,
        all_datasets_json=args.all_datasets_json,
        train=args.train,
        valid=args.valid,
        infer=args.infer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_caching_size=args.max_caching_size,
        dilated_model=args.dilated_model
    )


if __name__ == '__main__':
    _main()
