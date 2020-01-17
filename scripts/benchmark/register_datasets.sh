#!/usr/bin/env bash

#REAL_DATA="C:/ssd_data/RealTestSet_512"
#SYNTH_DATA="C:/ssd_data/SynthBarcode30k_512"

if [[ -z "$WORKDIR" ]]; then
    WORKDIR=.
fi
cd $WORKDIR

DATASETS_DIR="${WORKDIR}/logs/datasets"
mkdir -p $DATASETS_DIR
DATASETS_JSON_PATH="${DATASETS_DIR}/all_datasets.json"

# REAL DATASET
DATA_PATH="${REAL_DATA}"
DATASET_NAME="ZVZ-real"

DATASET_INFO_DIR="${DATA_PATH}/split"
SPLIT_DIR="${DATASET_INFO_DIR}/split_f9_t0,1,2,3,4_seed42"

for split in "_train" "_valid" "_infer" ""; do
    python data_preparation/update_datasets_json.py \
        --dataset_name "${DATASET_NAME}${split}" \
        --data_path $DATA_PATH \
        --csv_path "${SPLIT_DIR}/dataset${split}.csv" \
        --info_json_path "${DATASET_INFO_DIR}/dataset_info.json" \
        --datasets_json_path $DATASETS_JSON_PATH
done

# SYNTH DATASET
DATA_PATH_="${SYNTH_DATA}"
DATASET_NAME_="ZVZ-synth"

for i in $(seq -f "%02g" 1 10)
do
    DATA_PATH="${DATA_PATH_}/part${i}"
    DATASET_NAME="${DATASET_NAME_}${i}"
    DATASET_INFO_DIR="${DATA_PATH_}/split/part${i}"
    SPLIT_DIR="${DATASET_INFO_DIR}/split_f10_t0,1,2,3,4,5,6,7,8_seed42"
    for split in "_train" "_valid" ""; do
        python data_preparation/update_datasets_json.py \
            --dataset_name "${DATASET_NAME}${split}" \
            --data_path $DATA_PATH \
            --csv_path "${SPLIT_DIR}/dataset${split}.csv" \
            --info_json_path "${DATASET_INFO_DIR}/dataset_info.json" \
            --datasets_json_path $DATASETS_JSON_PATH
    done
done

DATA_PATH="${DATA_PATH_}"
DATASET_NAME="${DATASET_NAME_}"
DATASET_INFO_DIR="${DATA_PATH}/split/full"
SPLIT_DIR="${DATASET_INFO_DIR}"

for split in "_train" "_valid" ""; do
    python data_preparation/update_datasets_json.py \
        --dataset_name "${DATASET_NAME}${split}" \
        --data_path $DATA_PATH \
        --csv_path "${SPLIT_DIR}/dataset${split}.csv" \
        --info_json_path "${DATASET_INFO_DIR}/dataset_info.json" \
        --datasets_json_path $DATASETS_JSON_PATH
done
