#!/usr/bin/env bash

OUTPUT_PATH=$PWD"/test_output/*"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m trainer.train \
    --job-dir="output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --clusters=5 \
    --credentials_dir="" \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0., 0., 0.5],
      samples=200,
      drop_type=VANILLA,
      activation=tanh,
      cluster_layer=2"

rm -r "output_${NOW}"
