"""Main module for training model."""

import argparse
import logging
import os
import sys
from trainer.preprocess import load_data, _input_fn
from trainer.util import find_best_clustering, rand_index, match_proportion, adjusted_rand_index
from trainer.logging_util import append_spread_sheet

import tensorflow as tf
import numpy as np

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
    '--clusters',
    type=int,
    default=None,
  )

  parser.add_argument(
    '--spreadsheet_id',
    type=str,
    default='15li_sYrdZkyIpAHc404P8u3lLV6uj-Zjxyk--_V_ALA',
  )

  parser.add_argument(
    '--credentials_dir',
    type=str,
    default='',
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

  metrics_0 = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  tf.logging.info("train_and_evaluate: {}".format(metrics_0))

  metrics = estimator.evaluate(eval_input, steps=args.eval_steps)
  tf.logging.info("evaluate: {}".format(metrics))

  weight = estimator.get_variable_value('ff_net/ff' + str(hparams.cluster_layer) + '/kernel').T
  weight_dense = estimator.get_variable_value('ff_net/ff' + str(hparams.cluster_layer+1) + '/kernel')
  score, cluster1 = find_best_clustering(weight, clusters=args.clusters)
  score_next, cluster2 = find_best_clustering(weight_dense, clusters=args.clusters)
  score_scaled = score / np.sqrt(weight.shape[1])
  score_next_scaled = score_next / np.sqrt(weight_dense.shape[1])

  weight_mean = np.mean(np.abs(weight), axis=0)
  weight_next_mean = np.mean(np.abs(weight_dense), axis=0)

  r_index = rand_index(cluster1, cluster2)
  ar_index = adjusted_rand_index(cluster1, cluster2)
  match_prop = match_proportion(cluster1, cluster2)

  tf.logging.info("Cluster Score: {}".format(score))
  tf.logging.info("Cluster Score (scaled by dim): {}".format(score_scaled))
  tf.logging.info("Weight Mean Vector: {}".format(weight_mean))
  tf.logging.info("Next Cluster Score: {}".format(score_next))
  tf.logging.info("Next Cluster Score (scaled by dim): {}".format(score_next_scaled))
  tf.logging.info("Next Weight Mean Vector: {}".format(weight_next_mean))
  tf.logging.info("Cluster Matrix: {}".format(weight))
  tf.logging.info("Cluster Next Matrix: {}".format(weight_dense))
  tf.logging.info("Rand Index: {}".format(r_index))
  tf.logging.info("Adjusted Rand Index: {}".format(ar_index))
  tf.logging.info("Match Proportion: {}".format(match_prop))

  # Google Spreadsheet
  if args.spreadsheet_id is not None:
      norm_loss = metrics['norm_loss']
      sample_loss = metrics['sample_loss']
      norm_accuracy = metrics['norm_accuracy']
      sample_accuracy = metrics['sample_accuracy']
      sample_std = metrics['sample_std']

      record = [args.job_dir, args.batch_size, args.train_steps]
      hparams_name = ['samples', 'hidden_units', 'dropouts', 'activation', 'classes', 'type', 'drop_type', 'cluster_layer'
        , 'learning_rate', 'decay_step', 'decay_rate']
      for name in hparams_name:
          record.append(hparams.get(name))

      record += [norm_loss, sample_loss, norm_accuracy, sample_accuracy,
        sample_std, score, score_scaled, score_next, score_next_scaled,
        r_index, ar_index, match_prop, weight_mean, weight_next_mean,
        weight, weight_dense]
      append_spread_sheet(args.spreadsheet_id, record, args.credentials_dir, args.cloud_train)



if __name__ == "__main__":
  main()
