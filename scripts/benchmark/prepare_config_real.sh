#!/usr/bin/env bash

if [[ -z "$WORKDIR" ]]; then
    WORKDIR=.
fi
cd $WORKDIR

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

#TRAIN="ZVZ-real_train"  # registered_dataset_name
#VALID="ZVZ-real_valid"  # registered_dataset_name
# as many as you want, separated by space (may also be "")
INFER="ZVZ-real_infer ZVZ-real_valid ZVZ-real_train ZVZ-real"
if [[ -z "$TRAIN" ]]; then
    TRAIN=""
fi
if [[ -z "$VALID" ]]; then
    VALID=""
fi

# config preparation
CONFIG_IN_TEMPLATE="${WORKDIR}/configs/_template.yml"
CONFIG_CLASS="${WORKDIR}/configs/class_trees/barcode/base.yml"

# name of the dataset in json mapping of all datasets
if [[ -z "$DATASETS_JSON_PATH" ]]; then
    DATASETS_JSON_PATH="logs/datasets/all_datasets.json"
fi

for USE_DILATED_MODEL in "True" ""  # True, False
do
    if [[ "${USE_DILATED_MODEL}" == "True" ]]; then MODEL="dilated_model"; else MODEL="resnet18_unet"; fi
    CONFIG_DIR="${WORKDIR}/configs/generated/ZVZ-real/${MODEL}"
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
        --max_caching_size $MAX_CACHING_SIZE \
        --dilated_model="${USE_DILATED_MODEL}"
done
