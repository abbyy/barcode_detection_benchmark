"""
Updates json with registered datasets with provided arguments
(should be used to register additional datasets which was a result of train/valid/infer split)
"""
from argparse import ArgumentParser

from make_dataset import update_datasets_json_file

parser = ArgumentParser()
parser.add_argument(
    '--dataset_name', type=str, required=True
)
parser.add_argument(
    '--data_path', type=str, required=True
)
parser.add_argument(
    '--csv_path', type=str, required=True
)
parser.add_argument(
    '--info_json_path', type=str, required=True
)
parser.add_argument(
    '--datasets_json_path', type=str, required=True
)

if __name__ == '__main__':
    args = parser.parse_args()
    update_datasets_json_file(
        json_filename=args.datasets_json_path,
        update_key=args.dataset_name,
        data_path=args.data_path,
        csv_path=args.csv_path,
        info_json_path=args.info_json_path
    )
