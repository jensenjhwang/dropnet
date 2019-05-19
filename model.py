import tensorflow as tf
import tensorflow.keras as keras

NORMAL = 'normal'
SAMPLE = 'sample'

class FFNet(keras.Model):

    def __init__(self, mode, hparams):
        super(FFNet, self).__init__()
        self.mode= mode
        self.samples = hparams.samples
        feed_forward = keras.models.Sequential()
        for unit, d_prob in zip(hparams.hidden_units, hparams.dropouts):
            feed_forward.add(keras.layers.Dense(units=unit,activation=hparams.activation))
            if mode == tf.estimator.ModeKeys.TRAIN and hparams.type == NORMAL:
                feed_forward.add(keras.layers.Dropout(d_prob))
        feed_forward.add(keras.layers.Dense(units=hparams.classes, activation=hparams.activation))
        self.feed_forward = feed_forward

    def call(self, inputs):
        if self.mode == tf.estimator.ModeKeys.TRAIN or hparams.type == NORMAL:
            logits = self.feed_forward(inputs)
            return logits
        else:
            outputs = []
            for _ in hparams.samples:
                logits = self.feed_forward(inputs)
                outputs.append(logits)
            combined = outputs.concat(outputs, axis=-1)
            return combined

    @classmethod
    def get_tiny_hparams(cls):
        hparams = tf.contrib.training.HParams(
            samples=30,
            hidden_units=[20, 10, 10],
            dropouts=[0.5, 0, 0],
            activation='tanh',
            classes = 10,
            type=SAMPLE,
        )
        return hparams

    @classmethod
    def get_base_hparams(cls):
        hparams = tf.contrib.training.HParams(
            samples=30,
            hidden_units=[100, 100, 100],
            dropouts=[0.5, 0, 0],
            activation='tanh',
            classes = 10,
            type=SAMPLE,
        )
        return hparams

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  hparams = EncoderDecoder.get_base_hparams()
  hparams.add_hparam("learning_rate", 0.001)
  hparams.add_hparam("decay_step", 500)
  hparams.add_hparam("decay_rate", 0.9)
  return hparams

def model_fn(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: flattened pixels
    labels: one-hot labels
    mode: str. must be one of `tf.estimator.ModeKeys`.
    params: `tf.contrib.training.HParams` object containing hyperparameters for
      model.

  Returns:
    `tf.Estimator.EstimatorSpec` object.
  """
  training_hooks = []
  eval_hooks = []

  model = FFNet(mode, params)

  if mode == tf.estimator.ModeKeys.EVAL:

      model.type = NORMAL
      logits_norm = model(features)

      model.type = SAMPLE
      predictions = model(features)

      logits_sample = tf.math.reduce_mean(predictions, axis=-1)
      mean_std = tf.math.reduce_mean(tf.math.reduce_std(predictions, axis=-1))
      std_hook = tf.train.LoggingTensorHook(
        tensors={"mean_std": mean_std},
        every_n_iter=1
      )
      eval_hooks.append(std_hook)
      with tf.variable_scope("metrics"):
          eval_metric_ops = {
            "norm_accuracy": tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(input=logits_norm, axis=-1)),
            "sample_accuracy": tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(input=logits_sample, axis=-1))
          }

      model.type = params.type

      with tf.variable_scope("loss"):
          loss_norm = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_norm)
          loss_sample = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_sample)
          tf.summary.scalar("CE_loss_norm", loss_norm)
          tf.summary.scalar("CE_loss_sample", loss_sample)

          if model.type == NORMAL:
              loss = loss_norm
          else:
              loss = loss_sample

  else:
      logits = model(features)
      loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
      tf.summary.scalar("cross_entropy_loss", loss)

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=params.learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=params.decay_step,
      decay_rate=params.decay_rate,
      staircase=False,
    )
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predictions,
    eval_metric_ops=eval_metric_ops,
    training_hooks=training_hooks,
    evaluation_hooks=eval_hooks,
  )
