#!/usr/bin/env bash
# this script prepares your dataset to be used by this framework
#
# usage:
#
# DATA_DIR=... \
# DATASET_DIR=... \
# DATASET_NAME=... \
# NO_NEED_SPLIT=... \
# sh scripts/prepare_datasets.sh

# the following 4 env variables are the most important
# real_test_set
#DATA_DIR="C:/ssd_data/RealTestSet_512"
#DATASET_DIR="logs/datasets/real_test_512"
#DATASET_NAME="real_test_512"
#NO_NEED_SPLIT="false"

# path/to/dataset_root
if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="C:/ssd_data/dev_full/valid"
fi
# path/to/save/dataset
if [[ -z "$DATASET_DIR" ]]; then
    DATASET_DIR="logs/datasets/dev_full/valid"
fi
# name of the dataset in json mapping of all datasets
if [[ -z "$DATASET_NAME" ]]; then
    DATASET_NAME="default_dataset"
fi
# do we need to split dataset into train/valid?
if [[ -z "$NO_NEED_SPLIT" ]]; then
    NO_NEED_SPLIT="false"
fi

if [[ -z "$WORKDIR" ]]; then
    WORKDIR=.
fi
# name of the dataset in json mapping of all datasets
if [[ -z "$DATASETS_JSON_PATH" ]]; then
    DATASETS_JSON_PATH="${WORKDIR}/logs/datasets/all_datasets.json"
fi
# path/to/save/dataset_info
DATASET_INFO_DIR=$DATASET_DIR

python data_preparation/make_dataset.py \
    --dataset_path $DATA_DIR \
    --save_data_path $DATASET_DIR \
    --out_dataset $DATASET_INFO_DIR \
    --name $DATASET_NAME \
    --datasets_json $DATASETS_JSON_PATH

if [[ -z "$NO_NEED_SPLIT"  || "$NO_NEED_SPLIT" == "false" ]]; then
    N_FOLDS=10
#    N_FOLDS=2
    SEED=42
    TRAIN_FOLDS=0,1,2,3,4,5,6,7,8
#    TRAIN_FOLDS=0
    SPLIT="split_f${N_FOLDS}_t${TRAIN_FOLDS}_seed${SEED}"
    echo $SPLIT

    SPLIT_DIR=$DATASET_DIR/$SPLIT
    mkdir -p $SPLIT_DIR

    catalyst-data split-dataframe \
        --in-csv $DATASET_INFO_DIR/dataset.csv \
        --n-folds=$N_FOLDS --train-folds=$TRAIN_FOLDS \
        --out-csv=$SPLIT_DIR/dataset.csv

    python data_preparation/update_datasets_json.py \
        --dataset_name "${DATASET_NAME}_train" \
        --data_path $DATA_DIR \
        --csv_path $SPLIT_DIR/dataset_train.csv \
        --info_json_path $DATASET_INFO_DIR/dataset_info.json \
        --datasets_json_path $DATASETS_JSON_PATH

    python data_preparation/update_datasets_json.py \
        --dataset_name "${DATASET_NAME}_valid" \
        --data_path $DATA_DIR \
        --csv_path $SPLIT_DIR/dataset_valid.csv \
        --info_json_path $DATASET_INFO_DIR/dataset_info.json \
        --datasets_json_path $DATASETS_JSON_PATH
fi
