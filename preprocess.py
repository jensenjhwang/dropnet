
import tensorflow as tf
mnist = tf.keras.datasets.mnist

one_hot = True
batch_size = 16

def load_data():
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train = tf.convert_to_tensor(x_train)
	y_train = tf.convert_to_tensor(y_train)
	x_test = tf.convert_to_tensor(x_test)
	y_test = tf.convert_to_tensor(y_test)

	if one_hot:
		y_train = tf.one_hot(y_train, 10)
		y_test = tf.one_hot(y_test, 10)
	
	return (x_train, y_train),(x_test, y_test)

# Features and labels are tensors with equal first dims
def train_input_fn(features, labels, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	dataset = dataset.batch(batch_size)
	return dataset

def demo():
	train, test = load_data()
	x_train, y_train = train
	dataset = train_input_fn(x_train, y_train, batch_size)
	it = dataset.make_one_shot_iterator()
	next_element = it.get_next()

	num_batches = int(x_train.shape[0]) // batch_size

	with tf.Session() as sess:
		for i in range(num_batches):
			x_train, y_train = sess.run(next_element)

demo()