import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from model import FFNet, make_hparams
from preprocess import load_data, _input_fn

SAMPLE_SIZE = 10000
BATCH_SIZE = 16
NUM_EPOCHS = 1
PRINT_EVERY = 1

def get_preds(params, K):
	""" TODO: COMMENT
	"""
	(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
	predictions = []
	accuracies = []
	x_train = np.reshape(x_train, (x_train.shape[0], -1))
	x_test = tf.convert_to_tensor(np.reshape(x_test, (x_test.shape[0], -1)), dtype=tf.float32)

	sess = tf.Session()

	# y_test = tf.one_hot(tf.convert_to_tensor(y_test, dtype=tf.int32), 10)
	for i in range(K):
		idx = np.random.choice(len(x_train), SAMPLE_SIZE)
		x = tf.convert_to_tensor(x_train[idx], dtype=tf.float32)
		y = tf.one_hot(tf.convert_to_tensor(y_train[idx]), 10)
		
		# Train on x and y
		model = FFNet(tf.estimator.ModeKeys.TRAIN, params)
		x_j, y_j = _input_fn(x, y, BATCH_SIZE, True, NUM_EPOCHS)
		logits = model(x_j)
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_j)
		train_op = tf.train.GradientDescentOptimizer(params.learning_rate).minimize(loss)
		acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_j, axis=-1),
			predictions=tf.argmax(logits, axis=-1))

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		for j in range(NUM_EPOCHS):
			total_loss = 0
			accuracy = 0
			num_batches = SAMPLE_SIZE // BATCH_SIZE
			for k in range(num_batches):
				loss_value, _, _, _ = sess.run([loss, train_op, acc_op, acc])
				acc_value = sess.run(acc)
				total_loss += np.sum(loss_value)
				accuracy += acc_value / num_batches

			if j % PRINT_EVERY == 0:
				print('Epoch {}: loss = {}, accuracy = {}'.format(j, total_loss, accuracy))

		# Test
		model.mode = tf.estimator.ModeKeys.EVAL
		logits_test = tf.math.reduce_mean(model.call(x_test), axis=-1)

		logits_value = sess.run(logits_test)
		preds = np.argmax(logits_value, axis=-1)
		acc = np.mean(preds == y_test)
		accuracies.append(acc)
		predictions.append(logits_value)
		print('Test accuracy: {}'.format(acc))
	return predictions, accuracies

def getVariance(K):
	preds, accs = get_preds(make_hparams(), K)
	preds = np.stack(preds, axis=-1)
	var = np.mean((preds - np.mean(preds, axis=-1, keepdims=True)) ** 2)
	print('Variance is {}'.format(var))

getVariance(10)
