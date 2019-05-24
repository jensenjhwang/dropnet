#!/bin/bash

OUTPUT_PATH=$PWD"/test_output/*"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=30,
      drop_type=INVERTED,
      activation=tanh"
