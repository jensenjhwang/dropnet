#!/bin/bash

OUTPUT_PATH=$PWD"/test_output/none/"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0.2, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"

#!/bin/bash

OUTPUT_PATH=$PWD"/test_output/"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0.2, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=5 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"

#!/bin/bash

OUTPUT_PATH=$PWD"/test_output/"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0.2, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0, 0.2],
      samples=30,
      drop_type=VANILLA,
      activation=none"

#!/bin/bash

OUTPUT_PATH=$PWD"/test_output/"
NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0.2, 0.2, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0.2, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"


NOW=$(date '+%d_%m_%Y_%H_%M_%S')

# python command
python3 -m variance.variance \
    --job-dir="${OUTPUT_PATH}/output_${NOW}" \
    --train_steps=10 \
    --batch_size=16 \
    --hparams="hidden_units=[100, 100, 100],
      dropouts=[0, 0, 0.2],
      samples=100,
      drop_type=VANILLA,
      activation=none"


