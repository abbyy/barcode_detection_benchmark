#!/usr/bin/env bash

if [[ -z "$WORKDIR" ]]; then
    # hack to run in pycharm
    WORKDIR="F:/wd/odvss/"
    cd $WORKDIR
fi

if [[ -z "$NUM_WORKERS" ]]; then
#    NUM_WORKERS=0  # windows lovers
    NUM_WORKERS=4
fi
if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=4
fi
if [[ -z "$NUM_EPOCHS" ]]; then
    NUM_EPOCHS=100
fi
if [[ -z "$IMAGE_SIZE" ]]; then
    IMAGE_SIZE=512
fi
if [[ -z "$MAX_CACHING_SIZE" ]]; then
    MAX_CACHING_SIZE=1000
fi

TRAIN="synth_train"  # registered_dataset_name
VALID="synth_valid"  # registered_dataset_name
# as many as you want, separated by space (may also be "")
INFER="real_test real_test_valid \
synth01_valid synth02_valid synth03_valid \
synth04_valid synth05_valid synth06_valid \
synth07_valid synth08_valid synth09_valid \
synth10_valid synth_train synth_valid"

# config preparation
CONFIG_DIR="${WORKDIR}/configs/generated/c1"
CONFIG_IN_TEMPLATE="${WORKDIR}/configs/_template.yml"
CONFIG_CLASS="${WORKDIR}/configs/class_trees/barcode/base.yml"

# name of the dataset in json mapping of all datasets
if [[ -z "$DATASETS_JSON_PATH" ]]; then
    DATASETS_JSON_PATH="logs/datasets/all_datasets.json"
fi

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
