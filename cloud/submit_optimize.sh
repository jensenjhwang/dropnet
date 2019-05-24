#!/usr/bin/env bash

BASE_NAME=train

BUCKET=gs://dropnet

NOW=$(date '+%d_%m_%Y_%H_%M_%S')
JOB_NAME=${BASE_NAME}_${NOW}
JOB_DIR=${BUCKET}"/"${JOB_NAME}

STAGING_BUCKET=$BUCKET

OUTPUT_PATH=$JOB_DIR

MODULE_NAME=trainer.train

# Train on Cloud.
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --staging-bucket $STAGING_BUCKET \
    --module-name $MODULE_NAME \
    --package-path trainer/ \
    --config cloud/config_hparameter_optimization.yaml \
    -- \
    --train_steps=7500 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.6, 0.6, 0.6],
      samples=50" \
