# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        mnist_test 
# Date:             1/17/18 6:14 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/LearnTensorflow
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import argparse
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
'''
28 28 1
24 24 32
12 12 32
8 8 64
4 4 64
batch_size 256
batch_size 120
batch_size 84
batch_size 10
1.softmax就是激活函数，之前不需要再激活
2.变量初始化很重要，特别是w
3.样本之前要搅拌充分，否则会出现loss上下震荡
'''


class Net:
	input_image, input_label = [None] * 2
	w_1, w_2, w_3, w_4, w_5 = [None] * 5
	result_1_conv, result_2_conv, result_3_conv, result_4_conv = [None] * 4
	b_1, b_2, b_3, b_4, b_5 = [None] * 5
	result_1_relu, result_2_relu, result_3_relu, result_4_relu, result_5_relu = [None] * 5
	result_3_dropout, result_4_dropout, result_5_dropout = [None] * 3
	result_3_fc, result_4_fc, result_5_fc = [None] * 3
	result_1_maxpool, result_2_maxpool = [None] * 2
	result_expand = None
	loss, loss_mean = [None] * 2
	train_step = None
	correct_prediction = None
	accuracy = None
	keep_prob = None
	input_image_2 = None

	def __init__(self):
		print('new network')

	W_conv1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, IMG_CHANNEL, 32], dtype=tf.float32), name='w_1')
	def set_up_network(self, batch_size):
		# input and output
		with tf.name_scope('input'):
			self.input_image = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, IMG_WIDE * IMG_WIDE * IMG_CHANNEL], name='input_images'
			)
			self.input_image_2 = tf.reshape(self.input_image, [-1, 28, 28, 1])
			self.input_label = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, CLASS_NUM], name='input_labels'
			)
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

		# layer 1
		with tf.name_scope('layer_1'):
			# self.w_1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, IMG_CHANNEL, 32], dtype=tf.float32), name='w_1')
			self.w_1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, IMG_CHANNEL, 32], stddev=0.1, dtype=tf.float32))
			self.b_1 = tf.Variable(initial_value=tf.random_normal(shape=[32], dtype=tf.float32), name='b_1')
			self.result_1_conv = tf.nn.conv2d(
				input=self.input_image_2, filter=self.w_1,
				strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
			self.result_1_relu = tf.nn.relu(tf.nn.bias_add(self.result_1_conv, self.b_1), name='relu_1')
			self.result_1_maxpool = tf.nn.max_pool(
				value=self.result_1_relu, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME', name='maxpool_1')
		# self.result_1 = tf.nn.lrn(input=self.result_1_maxpool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		# self.result_1 = tf.nn.dropout(x=self.result_1, keep_prob=self.dropout)

		# layer 2
		with tf.name_scope('layer_2'):
			# self.w_2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64], dtype=tf.float32), name='w_2')
			self.w_2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
			self.b_2 = tf.Variable(initial_value=tf.random_normal(shape=[64], dtype=tf.float32), name='b_2')
			self.result_2_conv = tf.nn.conv2d(
				input=self.result_1_maxpool, filter=self.w_2,
				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
			self.result_2_relu = tf.nn.relu(tf.add(self.result_2_conv, self.b_2), name='relu_2')
			self.result_2_maxpool = tf.nn.max_pool(
				value=self.result_2_relu, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME', name='maxpool_2')
		# self.result_2 = tf.nn.lrn(input=self.result_2_maxpool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		# self.result_2 = tf.nn.dropout(x=self.result_2, keep_prob=self.dropout)

		# expand
		with tf.name_scope('expand'):
			self.result_expand = tf.reshape(self.result_2_maxpool, [-1, 7 * 7 * 64])
			print(self.result_expand)

		# layer 3
		with tf.name_scope('layer_3'):
			# self.w_3 = tf.Variable(initial_value=tf.random_normal(shape=[3136, 1024], dtype=tf.float32), name='w_4')
			self.w_3 = tf.Variable(initial_value=tf.truncated_normal(shape=[3136, 1024], stddev=0.1, dtype=tf.float32))
			self.b_3 = tf.Variable(initial_value=tf.random_normal(shape=[1024], dtype=tf.float32), name='b_4')
			self.result_3_fc = tf.matmul(self.result_expand, self.w_3, name='result_4_multiply')
			self.result_3_relu = tf.nn.relu(tf.add(self.result_3_fc, self.b_3), name='relu_4')
			# self.result_3_dropout = self.result_3_relu
			self.result_3_dropout = tf.nn.dropout(x=self.result_3_relu, keep_prob=self.keep_prob, name='dropout_4')

		# layer 4
		with tf.name_scope('layer_4'):
			# self.w_4 = tf.Variable(initial_value=tf.random_normal(shape=[1024, 10], dtype=tf.float32), name='w_4')
			self.w_4 = tf.Variable(initial_value=tf.truncated_normal(shape=[1024, 10], stddev=0.1, dtype=tf.float32))
			self.b_4 = tf.Variable(initial_value=tf.random_normal(shape=[10], dtype=tf.float32), name='b_4')
			self.result_4_fc = tf.nn.bias_add(tf.matmul(self.result_3_dropout, self.w_4), self.b_4, name='result_4_fc')
			# self.result_4_relu = tf.nn.relu(self.result_4_fc, name='result_4_relu')
			self.result_4_relu = self.result_4_fc
			# self.result_4_dropout = tf.nn.dropout(x=self.result_4_relu, keep_prob=self.keep_prob, name='dropout_4')
			# self.result_4_dropout = self.result_4_relu

		# layer 5
		with tf.name_scope('layer_5'):
		# 	self.w_5 = tf.Variable(initial_value=tf.random_normal(shape=[84, 10], dtype=tf.float32), name='w_5')
		# 	self.b_5 = tf.Variable(initial_value=tf.random_normal(shape=[10], dtype=tf.float32), name='b_5')
		# 	self.result_5_fc = tf.add(tf.matmul(self.result_4_dropout, self.w_5), self.b_5, name='result_5_fc')
		# 	# self.result_5_relu = tf.nn.relu(self.result_5_fc, name='result_5_relu')
		# 	self.result_5_relu = self.result_5_fc
			self.result_5_relu = self.result_4_relu

		# softmax loss
		with tf.name_scope('softmax_loss'):
			self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.result_5_relu, name='loss')
			self.loss_mean = tf.reduce_mean(self.loss)

		# accuracy
		with tf.name_scope('accuracy'):
			self.correct_prediction = tf.equal(tf.argmax(self.result_5_relu, 1), tf.argmax(self.input_label, 1))
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		# Gradient Descent
		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_mean)

	def train_the_model(self, train_files_queue = ''):
		# check point
		mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
		tf.summary.scalar("loss", self.loss_mean)
		tf.summary.scalar('accuracy', self.accuracy)
		merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			for i in range(400):
				batch = mnist.train.next_batch(TRAIN_BATCH_SIZE)
				# print(batch[1])
				train_accuracy, train_loss, summary_str = sess.run(
					[self.accuracy, self.loss_mean, merged_summary],
					feed_dict={self.input_image: batch[0], self.input_label: batch[1], self.keep_prob: 1.0}
				)
				# train_writer.add_summary(summary_str, i)
				if i % 10 == 0:
					# train_accuracy = accuracy.eval(feed_dict={
					# 	x: batch[0], y_: batch[1], keep_prob: 1.0})
					print('step %d, training loss %g, training accuracy %g' % (i, train_loss,train_accuracy))
				# train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
				sess.run([self.train_step], feed_dict={self.input_image: batch[0], self.input_label: batch[1], self.keep_prob: 0.8})


def main():
	net = Net()
	net.set_up_network(TRAIN_BATCH_SIZE)
	net.train_the_model()


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
	main()
	# tf.app.run(main=main)   # , argv=[sys.argv[0]] + unparsed
