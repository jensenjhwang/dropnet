#!/usr/bin/env bash

OUTPUT_PATH=$PWD"/test_output/*"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m trainer.train \
    --job-dir="output_${NOW}" \
    --train_steps=7500 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=200,
      drop_type=INVERTED,
      activation=tanh"
