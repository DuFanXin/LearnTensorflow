# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        mnist 
# Date:             1/16/18 2:10 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/LearnTensorflow
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import argparse
import sys
# import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

IMG_WIDE, IMG_HEIGHT, IMG_CHANNEL = 28, 28, 1
EPOCH_NUM = 10
TRAIN_BATCH_SIZE = 50
DEVELOPMENT_BATCH_SIZE = 10
TEST_BATCH_SIZE = 128
EPS = 10e-5
FLAGS = None
CLASS_NUM = 10
FLAGS = None


def deepnn(x):
	# Reshape to use within a convolutional neural net.
	# Last dimension is for "features" - there is only one here, since images are
	# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1])

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	with tf.name_scope('conv1'):
		# W_conv1 = weight_variable([5, 5, 1, 32])
		# w_1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, IMG_CHANNEL, 32], dtype=tf.float32), name='w_1')
		w_1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, IMG_CHANNEL, 32], stddev=0.1, dtype=tf.float32), name='w_1')
		b_1 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[32], dtype=tf.float32), name='b_1')
		# h_conv1 = tf.nn.relu(conv2d(x_image, w_1) + b_1)
		result_1_conv = tf.nn.conv2d(
			input=x_image, filter=w_1,
			strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
		result_1_relu = tf.nn.relu(tf.nn.bias_add(result_1_conv, b_1), name='relu_1')
		print(result_1_relu)

	# Pooling layer - downsamples by 2X.
	with tf.name_scope('pool1'):
		# h_pool1 = max_pool_2x2(result_1_relu)
		result_1_maxpool = tf.nn.max_pool(
			value=result_1_relu, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME', name='maxpool_1')
		print(result_1_maxpool)

	# Second convolutional layer -- maps 32 feature maps to 64.
	with tf.name_scope('conv2'):
		# W_conv2 = weight_variable([5, 5, 32, 64])
		# w_2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64], dtype=tf.float32), name='w_2')
		# w_2 = weight_variable([5, 5, 32, 64])
		w_2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32), name='w_2')
		b_2 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[64], dtype=tf.float32), name='b_2')
		result_2_conv = tf.nn.conv2d(
			input=result_1_maxpool, filter=w_2,
			strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
		result_2_relu = tf.nn.relu(tf.add(result_2_conv, b_2), name='relu_2')
		# b_conv2 = bias_variable([64])
		# h_conv2 = tf.nn.relu(conv2d(result_1_maxpool, w_2) + b_2)
		print(result_2_relu)

	# Second pooling layer.
	with tf.name_scope('pool2'):
		# h_pool2 = max_pool_2x2(result_2_relu)
		result_2_maxpool = tf.nn.max_pool(
			value=result_2_relu, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME', name='maxpool_2')
		print(result_2_maxpool)

	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('fc1'):
		w_3 = weight_variable([7 * 7 * 64, 1024])
		# w_3 = tf.Variable(initial_value=tf.random_normal(shape=[3136, 1024], dtype=tf.float32), name='w_4')
		b_3 = tf.Variable(initial_value=tf.random_normal(shape=[1024], dtype=tf.float32), name='b_4')
		# b_3 = bias_variable([1024])
		'''问题所在，变量初始化有问题，他的效果比较好'''
		# h_pool2_flat = tf.reshape(result_2_maxpool, [-1, 7*7*64])
		result_expand = tf.reshape(result_2_maxpool, [-1, 7 * 7 * 64])
		# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		result_3_fc = tf.matmul(result_expand, w_3, name='result_4_multiply')
		result_3_relu = tf.nn.relu(tf.add(result_3_fc, b_3), name='relu_4')

	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		# h_fc1_drop = tf.nn.dropout(result_3_relu, keep_prob)
		result_3_dropout = tf.nn.dropout(x=result_3_relu, keep_prob=keep_prob, name='dropout_4')

	# Map the 1024 features to 10 classes, one for each digit
	with tf.name_scope('fc2'):
		w_4 = weight_variable([1024, 10])
		b_4 = bias_variable([10])
		# y_conv = tf.matmul(result_3_dropout, w_4) + b_2
		# w_4 = tf.Variable(initial_value=tf.truncated_normal(shape=[1024, 10], stddev=0.1, dtype=tf.float32))
		# b_4 = tf.Variable(initial_value=tf.random_normal(shape=[10], dtype=tf.float32), name='b_4')
		result_4_fc = tf.nn.bias_add(tf.matmul(result_3_dropout, w_4), b_4, name='result_4_fc')
		# result_4_relu = tf.nn.relu(result_4_fc, name='result_4_relu')
		result_4_relu = result_4_fc

	return result_4_relu, keep_prob


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])

	# Build the graph for the deep net
	y_conv, keep_prob = deepnn(x)

	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
	cross_entropy = tf.reduce_mean(cross_entropy)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)

	graph_location = FLAGS.tb_dir
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())
	tf.summary.scalar("loss", cross_entropy)
	tf.summary.scalar('accuracy', accuracy)
	merged_summary = tf.summary.merge_all()
	all_parameters_saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		for i in range(200):
			batch = mnist.train.next_batch(50)
			train_accuracy, train_loss, summary_str = sess.run(
				[accuracy, cross_entropy, merged_summary],
				feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}
			)
			train_writer.add_summary(summary_str, i)
			if i % 10 == 0:
				# train_accuracy = accuracy.eval(feed_dict={
				# 	x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			# train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

		# print('test accuracy %g' % accuracy.eval(feed_dict={
		# 	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# 数据地址
	parser.add_argument(
		'--data_dir', type=str, default='../input-data/MNIST',
		help='Directory for storing input data')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='../output-data/models',
		help='output model path')

	# 模型保存地址
	parser.add_argument(
		'--tb_dir', type=str, default='../output-data/log',
		help='TensorBoard log path')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main)   # , argv=[sys.argv[0]] + unparsed
