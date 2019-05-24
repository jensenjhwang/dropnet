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
    --train_steps=20000 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 500, 100],
      dropouts=[0., 0.5, 0.],
      samples=200,
      drop_type=VANILLA,
      activation=tanh" \
