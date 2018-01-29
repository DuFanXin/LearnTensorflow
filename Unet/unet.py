# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        unet 
# Date:             1/26/18 11:28 AM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/LearnTensorflow
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import tensorflow as tf
# import keras
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 572, 572, 3
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT = 387, 387
EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 128
DEVELOPMENT_BATCH_SIZE = 10
TEST_BATCH_SIZE = 128
EPS = 10e-5
FLAGS = None
CLASS_NUM = 10


class Unet:
	input_image = None
	input_label = None
	keep_prob = None
	lamb = None
	result_expand = None
	loss, loss_mean, train_step = [None] * 3
	prediction, correct_prediction, accuracy = [None] * 3
	result_conv = {}
	result_relu = {}
	result_maxpool = {}
	result_from_contract_layer = {}
	w = {}
	b = {}

	def __init__(self):
		print('New U-net Network')

	def init_w(self, shape, name):
		w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32), name=name)
		tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
		return w

	@staticmethod
	def init_b(shape, name):
		return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

	@staticmethod
	def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
		result_from_contract_layer_shape = tf.shape(result_from_contract_layer)
		result_from_upsampling_shape = tf.shape(result_from_upsampling)
		result_from_contract_layer_crop = \
			tf.slice(
				input_=result_from_contract_layer,
				begin=[
					0,
					(result_from_contract_layer_shape[1] - result_from_upsampling_shape[1]) // 2,
					(result_from_contract_layer_shape[2] - result_from_upsampling_shape[2]) // 2,
					0
				],
				size=[
					result_from_upsampling_shape[0],
					result_from_upsampling_shape[1],
					result_from_upsampling_shape[2],
					result_from_upsampling_shape[3]
				]
			)
		return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=3)

	def set_up_unet(self, batch_size):
		with tf.name_scope('input'):
			# learning_rate = tf.train.exponential_decay()
			self.input_image = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, INPUT_IMG_WIDE, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL], name='input_images'
			)

			# for softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# using one-hot
			# self.input_label = tf.placeholder(
			# 	dtype=tf.float32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE, CLASS_NUM], name='input_labels'
			# )

			# for sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# not using one-hot coding
			self.input_label = tf.placeholder(
				dtype=tf.int64, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
			)
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
			self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')

		# layer 1
		with tf.name_scope('layer_1'):
			# conv_1
			self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
			self.b[1] = self.init_b(shape=[64], name='b_1')
			result_conv_1 = tf.nn.conv2d(
				input=self.input_image, filter=self.w[1],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias'), name='relu_1')

			# conv_2
			self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
			self.b[2] = self.init_b(shape=[64], name='b_2')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[2],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[2], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[1] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')
			# self.result_1 = tf.nn.lrn(input=self.result_1_maxpool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
			# self.result_1 = tf.nn.dropout(x=self.result_1, keep_prob=self.dropout)

		# layer 2
		with tf.name_scope('layer_2'):
			# conv_1
			self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
			self.b[3] = self.init_b(shape=[128], name='b_3')
			result_conv_1 = tf.nn.conv2d(
				input=result_maxpool, filter=self.w[3],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[3], name='add_bias'), name='relu_1')

			# conv_2
			self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
			self.b[4] = self.init_b(shape=[128], name='b_4')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[4],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[4], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[2] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

		# layer 3
		with tf.name_scope('layer_3'):
			# conv_1
			self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
			self.b[5] = self.init_b(shape=[256], name='b_5')
			result_conv_1 = tf.nn.conv2d(
				input=result_maxpool, filter=self.w[5],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[5], name='add_bias'), name='relu_1')

			# conv_2
			self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
			self.b[6] = self.init_b(shape=[256], name='b_6')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[6],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[6], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[3] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

		# layer 4
		with tf.name_scope('layer_4'):
			# conv_1
			self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
			self.b[7] = self.init_b(shape=[512], name='b_7')
			result_conv_1 = tf.nn.conv2d(
				input=result_maxpool, filter=self.w[7],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[7], name='add_bias'), name='relu_1')

			# conv_2
			self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
			self.b[8] = self.init_b(shape=[512], name='b_8')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[8],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[8], name='add_bias'), name='relu_2')
			self.result_from_contract_layer[4] = result_relu_2  # 该层结果临时保存, 供上采样使用

			# maxpool
			result_maxpool = tf.nn.max_pool(
				value=result_relu_2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

		# layer 5 (bottom)
		with tf.name_scope('layer_5'):
			# conv_1
			self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
			self.b[9] = self.init_b(shape=[1024], name='b_9')
			result_conv_1 = tf.nn.conv2d(
				input=result_maxpool, filter=self.w[9],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[9], name='add_bias'), name='relu_1')

			# conv_2
			self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
			self.b[10] = self.init_b(shape=[1024], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[10],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[10], name='add_bias'), name='relu_2')

			# up sample
			self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
			self.b[11] = self.init_b(shape=[512], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[11],
				output_shape=[batch_size, 56, 56, 512], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[11], name='add_bias'), name='relu_3')

		# layer 6
		with tf.name_scope('layer_6'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_relu_3)
			# print(result_merge)

			# conv_1
			self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
			self.b[12] = self.init_b(shape=[512], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[12],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[12], name='add_bias'), name='relu_1')

			# conv_2
			self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_10')
			self.b[13] = self.init_b(shape=[512], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[13],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[13], name='add_bias'), name='relu_2')

			# up sample
			self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_11')
			self.b[14] = self.init_b(shape=[256], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[14],
				output_shape=[batch_size, 104, 104, 256], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[14], name='add_bias'), name='relu_3')

		# layer 7
		with tf.name_scope('layer_7'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_relu_3)

			# conv_1
			self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_12')
			self.b[15] = self.init_b(shape=[256], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[15],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[15], name='add_bias'), name='relu_1')

			# conv_2
			self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_10')
			self.b[16] = self.init_b(shape=[256], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[16],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[16], name='add_bias'), name='relu_2')

			# up sample
			self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_11')
			self.b[17] = self.init_b(shape=[128], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[17],
				output_shape=[batch_size, 200, 200, 128], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[17], name='add_bias'), name='relu_3')

		# layer 8
		with tf.name_scope('layer_8'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_relu_3)

			# conv_1
			self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_12')
			self.b[18] = self.init_b(shape=[128], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[18],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[18], name='add_bias'), name='relu_1')

			# conv_2
			self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_10')
			self.b[19] = self.init_b(shape=[128], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[19],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[19], name='add_bias'), name='relu_2')

			# up sample
			self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_11')
			self.b[20] = self.init_b(shape=[64], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[20],
				output_shape=[batch_size, 392, 392, 64], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[20], name='add_bias'), name='relu_3')

		# layer 7
		with tf.name_scope('layer_7'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_relu_3)

			# conv_1
			self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_12')
			self.b[21] = self.init_b(shape=[64], name='b_12')
			result_conv_1 = tf.nn.conv2d(
				input=result_merge, filter=self.w[21],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[21], name='add_bias'), name='relu_1')

			# conv_2
			self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_10')
			self.b[22] = self.init_b(shape=[64], name='b_10')
			result_conv_2 = tf.nn.conv2d(
				input=result_relu_1, filter=self.w[22],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[22], name='add_bias'), name='relu_2')

			# convolution to [batch_size, IMG_WIDE, IMG_HEIGHT, CLASS_NUM]
			self.w[23] = self.init_w(shape=[2, 2, 64, CLASS_NUM], name='w_11')
			self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
			result_conv_3 = tf.nn.conv2d(
				input=result_relu_2, filter=self.w[23],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
			self.prediction = tf.nn.relu(tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='relu_2')
			# print(self.prediction)
			# print(self.input_label)

		# softmax loss
		with tf.name_scope('softmax_loss'):
			# using one-hot
			# self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')

			# not using one-hot
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			self.loss_mean = tf.reduce_mean(self.loss)
			tf.add_to_collection(name='loss', value=self.loss_mean)
			self.loss_mean = tf.add_n(inputs=tf.get_collection(key='loss'))

		# accuracy
		with tf.name_scope('accuracy'):
			# using one-hot
			# self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_label, 1))

			# not using one-hot
			self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3), self.input_label)
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		# Gradient Descent
		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_mean)


if __name__ == '__main__':
	net = Unet()
	net.set_up_unet(TRAIN_BATCH_SIZE)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		sess.run(net.train_step)

