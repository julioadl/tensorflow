import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import random
import numpy as np

def random_mini_batches(X_train, Y_train, minibatch_size):
	m, n_x = X_train.shape
	minibatches = []
	for i in range(int(m / minibatch_size)):
		random_i = random.randint(0, n_x)
		minibatch_X = X_train[:,random_i: random_i + minibatch_size]
		minibatch_Y = Y_train[:,random_i: random_i + minibatch_size]
		minibatches.append((minibatch_X, minibatch_Y))
	return minibatches

def one_hot_matrix(labels, C):
	C = tf.constant(C, name="C")
	one_hot_matrix = tf.one_hot(labels, C, axis=0)
	with tf.Session() as sess:
		one_hot = sess.run(one_hot_matrix)
	return one_hot

def sigmoid(z):
	x = tf.placeholder(tf.float32, name="x")
	sigmoid = tf.sigmoid(x)
	with tf.Session() as sess:
		result = sess.run(sigmoid, feed_dict = {x:z})

	return result

def cost(logits, labels):
	z = tf.placeholder(tf.float32, name="z")
	y = tf.placeholder(tf.float32, name="y")
	cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
	with tf.Session as sess:
		cost = sess.run(cost, feed_dict={z: logits, y:labels})

	return cost

def create_placeholders(n_x, n_y):
	X = tf.placeholder(dtype=tf.float32, shape=(n_x, None), name="X")
	Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None), name="Y")
	return X, Y

def initialize_parameters():
	W1 = tf.get_variable("W1", [25, 784], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [10, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b3 = tf.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())

	parameters = {"W1": W1,
					"b1": b1,
					"W2": W2,
					"b2": b2,
					"W3": W3,
					"b3": b3}

	return parameters

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)

	return Z3

def compute_cost(Z3, Y, weights):
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) + 0.1*tf.nn.l2_loss(weights))
	return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
	ops.reset_default_graph()
	seed = 3
	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []

	X, Y = create_placeholders(n_x, n_y)
	parameters = initialize_parameters()
	Z3 = forward_propagation(X, parameters)
	cost = compute_cost(Z3, Y, parameters["W3"])
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y:minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate = " + str(learning_rate))
		plt.show()

		parameters = sess.run(parameters)
		print "Parameters have been trained"

		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print "Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})
		print "Test Accuracy:", accuracy.eval({X: X_test, Y:Y_test})

		return parameters

if __name__ == "__main__":
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	X_train = mnist.train.images
	Y_train = np.asarray(mnist.train.labels, dtype=np.int32)
	X_test = mnist.test.images
	Y_test = np.asarray(mnist.test.labels, dtype=np.int32)
	X_train = X_train.T
	Y_train = one_hot_matrix(Y_train, 10)
	X_test = X_test.T
	Y_test = one_hot_matrix(Y_test, 10)
	parameters = model(X_train, Y_train, X_test, Y_test)