import tensorflow as tf
import numpy as np
from model import FFNet, make_hparams

IMG_SIZE = 28

def gradAscent(params, image):
	model = FFNet('DROPOUT', params)

	x = tf.placeholder(tf.float32, image.shape)
	x_hat = tf.get_variable("image", (1, IMG_SIZE * IMG_SIZE), dtype=tf.float32)
	assign_op = tf.assign(x_hat, x)
	lr = tf.placeholder(tf.float32, ())
	y_hat = tf.placeholder(tf.int32, ())
	labels = tf.one_hot(y_hat, 10)
	logits = tf.math.reduce_mean(model.call(x_hat), axis=-1)
	
	loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
	optim_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=[x_hat])
	epsilon = tf.placeholder(tf.float32, ())
	below = x - epsilon
	above = x + epsilon
	projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
	with tf.control_dependencies([projected]):
		project_step = tf.assign(x_hat, projected)

	eps0 = 2
	lr0 = 1e-2
	steps = 1000
	target0 = 8
	print_every = 100

	init_op = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(assign_op, feed_dict={x:image})
		sess.run(init_op)
		for i in range(steps):
			_, loss_value = sess.run([optim_step, loss],
				feed_dict={lr: lr0, y_hat: target0, epsilon: eps0})
			img = sess.run(x_hat)
			if i % print_every == 0:
				print(loss_value)

def main():
	hparams = make_hparams()
	image = np.zeros((1, IMG_SIZE * IMG_SIZE))
	gradAscent(hparams, image)

if __name__ == "__main__":
	main()