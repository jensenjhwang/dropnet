import tensorflow as tf


def load_data(one_hot=True):
	(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = tf.convert_to_tensor(x_train)
	y_train = tf.convert_to_tensor(y_train)
	x_test = tf.convert_to_tensor(x_test)
	y_test = tf.convert_to_tensor(y_test)

	if one_hot:
		y_train = tf.one_hot(y_train, 10)
		y_test = tf.one_hot(y_test, 10)

	return (x_train, y_train),(x_test, y_test)

# Features and labels are tensors with equal first dims
def _input_fn(features, labels, batch_size, shuffle):
	features = tf.cast(features,tf.float32)
	labels = tf.cast(labels,tf.int32)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	# Shuffle, repeat, and batch the examples.
	if shuffle:
		dataset = dataset.shuffle(1000)

	dataset = dataset.batch(batch_size)

	dataset_iterator = dataset.make_one_shot_iterator()

	return dataset_iterator.get_next()
