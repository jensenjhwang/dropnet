import tensorflow as tf
import tensorflow.keras as keras

NORMAL = 'normal'
SAMPLE = 'sample'

class FFNet(keras.Model):

    def __init__(self, mode, hparams):
        super(FFNet, self).__init__()
        self.mode= mode
        self.samples = hparams.samples
        self.type = hparams.type

        self.feed_forward = []
        for unit in hparams.hidden_units:
            self.feed_forward.append(keras.layers.Dense(units=unit,activation=hparams.activation))

        self.dropouts = []
        for d_prob in hparams.dropouts:
            self.dropouts.append(keras.layers.Dropout(rate=d_prob))

        self.postprocess = keras.layers.Dense(units=hparams.classes, activation=hparams.activation)

        # feed_forward = keras.models.Sequential()
        # training = (mode == tf.estimator.ModeKeys.TRAIN) or hparams.type == SAMPLE
        # for unit, d_prob in zip(hparams.hidden_units, hparams.dropouts):
        #     feed_forward.add(keras.layers.Dense(units=unit,activation=hparams.activation))
        #     feed_forward.add(tf.layers.dropout())
        #     # if mode == tf.estimator.ModeKeys.TRAIN or hparams.type == SAMPLE:
        #     #     feed_forward.add(keras.layers.Dropout(d_prob))
        # feed_forward.add(keras.layers.Dense(units=hparams.classes, activation=hparams.activation))
        # self.feed_forward = feed_forward

    def predict(self, inputs, training):
        result = inputs
        for unit, dropout in zip(self.feed_forward, self.dropouts):
            result = dropout(unit(result), training=training)
        result = self.postprocess(result)
        return result

    def call(self, inputs):

        if self.mode == tf.estimator.ModeKeys.TRAIN or self.type == NORMAL:
            logits = self.predict(inputs, self.mode==tf.estimator.ModeKeys.TRAIN)
            return logits
        else:
            outputs = []
            for _ in range(self.samples):
                logits = self.predict(inputs, True)
                outputs.append(logits)
            combined = tf.stack(outputs, axis=-1)
            return combined

        # if self.mode == tf.estimator.ModeKeys.TRAIN or self.type == NORMAL:
        #     logits = self.feed_forward(inputs)
        #     return logits
        # else:
        #
        #     outputs = []
        #     for _ in range(self.samples):
        #         logits = self.feed_forward(inputs)
        #         outputs.append(logits)
        #     combined = tf.stack(outputs, axis=-1)
        #     return combined

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
  hparams = FFNet.get_base_hparams()
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

  features = tf.reshape(features, [tf.shape(features)[0], features.shape[1]*features.shape[2]])

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
        every_n_iter=100
      )
      eval_hooks.append(std_hook)
      with tf.variable_scope("metrics"):
          eval_metric_ops = {
            "norm_accuracy": tf.metrics.accuracy(
                labels=tf.argmax(labels, axis=-1),
                predictions=tf.argmax(input=logits_norm, axis=-1)),
            "sample_accuracy": tf.metrics.accuracy(
                labels=tf.argmax(labels, axis=-1),
                predictions=tf.argmax(input=logits_sample, axis=-1))
          }

      model.type = params.type

      with tf.variable_scope("loss"):
          loss_norm = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_norm)
          loss_sample = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_sample)
          tf.summary.scalar("CE_loss_norm", loss_norm)
          tf.summary.scalar("CE_loss_sample", loss_sample)

          if model.type == NORMAL:
              loss = loss_norm
              logits = logits_norm
          else:
              loss = loss_sample
              logits = logits_sample

          loss_hook = tf.train.LoggingTensorHook(
            tensors={"CE_loss": loss},
            every_n_iter=100
          )
          eval_hooks.append(loss_hook)

      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=logits,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=eval_hooks,
      )

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
    predictions=logits,
    training_hooks=training_hooks,
  )
