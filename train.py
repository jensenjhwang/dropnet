"""Main module for training model."""

import argparse
import logging
import os
import sys
from trainer.preprocess import load_data, _input_fn

import tensorflow as tf

# Add `Dropnet` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trainer.model as model

_PREDICT="PREDICT"
_TRAIN="TRAIN"

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--job-dir',
    help='Path to save test_output of model including checkpoints.',
    type=str,
    required=True
  )

  parser.add_argument(
    '--hparams',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
  )

  parser.add_argument(
    '--warm_start_from_dir',
    type=str,
    help='Checkpoint directory to warm start from.',
    default=''
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
  )

  parser.add_argument(
    '--train_steps',
    type=int,
    required=True
  )

  parser.add_argument(
    '--eval_steps',
    type=int,
    default=100,
  )

  parser.add_argument(
    '--profile_steps',
    type=int,
    default=200,
  )

  parser.add_argument(
    '--save_checkpoint_steps',
    type=int,
    default=500,
  )

  parser.add_argument(
    '--log_step_count',
    type=int,
    default=200,
  )

  parser.add_argument(
    '--num_splits',
    type=int,
    default=10
  )
  parser.add_argument{}

  ### Model HParams
  parser.add_argument(
    '--learning_rate',
    type=float,
  )
  parser.add_argument(
    '--decay_rate',
    type=float,
  )
  parser.add_argument(
    '--samples',
    type=int,
  )
  parser.add_argument(
    '--hidden_units',
    nargs='+',
    type=int,
  )
  parser.add_argument(
    '--dropouts',
    nargs='+',
    type=float,
  )
  parser.add_argument(
    '--activation',
    type=str,
  )
  parser.add_argument(
    '--type',
    type=str,
  )

  args, _ = parser.parse_known_args()

  return args


def set_up_logging():
  """Sets up logging."""

  # Check for environmental variable.
  file_location = os.getenv('JOB_DIRECTORY', '.')

  print("Logging file writing to {}".format(file_location), flush=True)

  logging.basicConfig(
    filename=os.path.join(file_location, 'training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(process)d - %(message)s'
  )

  logging.debug("Initialize debug.")

def main():

  args = parse_args()

  set_up_logging()

  (x_train, y_train),(x_test, y_test) = load_data()

  train_input = lambda: _input_fn(x_train, y_train, args.batch_size, shuffle=True)
  eval_input = lambda: _input_fn(x_test, y_test, args.batch_size, shuffle=False)

  hparams = model.make_hparams()
  hparams.parse(args.hparams)

  logging.info("HParams {}".format(hparams))

  # Optionally warm start variables.
  if args.warm_start_from_dir != '':
    warm_start_from = tf.estimator.WarmStartSettings(
      args.warm_start_from_dir,
    )
  else:
    warm_start_from=None

  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
    model_dir=args.job_dir,
    log_step_count_steps=args.log_step_count,
    save_checkpoints_steps=args.save_checkpoint_steps
  )

  estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    config=run_config,
    params=hparams,
    warm_start_from=warm_start_from
  )

  # Hook to log step timing.
  hook = tf.train.ProfilerHook(
    save_steps=args.profile_steps,
    output_dir=args.job_dir,
    show_memory=True
  )

  logging.info("Defining `train_spec`.")
  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input,
    max_steps=args.train_steps,
    hooks=[hook]
  )

  logging.info("Defining `eval_spec`.")
  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input,
    steps=args.eval_steps,
    throttle_secs=5
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
  main()
