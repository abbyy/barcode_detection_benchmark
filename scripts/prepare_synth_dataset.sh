#!/usr/bin/env bash

for i in $(seq -f "%02g" 1 10)
do
    DATA_DIR="C:/ssd_data/SynthBarcode30k_512/part$i" \
    DATASET_DIR="logs/datasets/SynthBarcode30k_512/part$i" \
    DATASET_NAME="synth$i" \
    NO_NEED_SPLIT="false" \
    bash scripts/prepare_dataset.sh
done

DATASETS_JSON_PATH="logs/datasets/all_datasets.json"
DATASET_DIR="logs/datasets/SynthBarcode30k_512/full"

python data_preparation/merge_datasets.py \
    --aggregated_name "synth" \
    --datasets_json $DATASETS_JSON_PATH \
    --out_dataset_dir $DATASET_DIR
