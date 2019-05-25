#!/usr/bin/env bash

BASE_NAME=train

BUCKET=gs://dropnet

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name trainer.train\
    --package-path trainer/ \
    --config cloud/config_gpu.yaml \
    -- \
    --train_steps=2500 \
    --batch_size=128 \
    --credentials_dir="gs://dropnet" \
    --cloud_train \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.8, 0., 0.],
      samples=200,
      drop_type=INVERTED,
      activation=tanh,
      cluster_layer=0"
