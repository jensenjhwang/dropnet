#!/usr/bin/env bash

OUTPUT_PATH=$PWD"/test_output/*"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 train.py \
    --job-dir="output_${NOW}" \
    --train_steps=1000 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.5, 0.5, 0.5],
      samples=100"
