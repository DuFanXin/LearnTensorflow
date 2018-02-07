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
import argparse
import os
# import keras
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 284, 284, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 100, 100, 1
EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 1
DEVELOPMENT_BATCH_SIZE = 10
TEST_BATCH_SIZE = 128
EPS = 10e-5
FLAGS = None
CLASS_NUM = 2
'''
类别名称     R G B
background  0 0 0       背景
aeroplane   128 0 0     飞机
bicycle     0 128 0     自行车
bird        128 128 0   鸟
boat        0 0 128     船
bottle      128 0 128   瓶子
bus         0 128 128   大巴
car         128 128 128 车
cat         64 0 0      猫
chair       192 0 0     椅子
cow         64 128 0    牛
diningtable 192 128 0   餐桌
dog         64 0 128    狗
horse       192 0 128   马
motorbike   64 128 128  摩托车
person      192 128 128 人
pottedplant 0 64 0      盆栽
sheep       128 64 0    羊
sofa        0 192 0     沙发
train       128 192 0   火车
tvmonitor   0 64 128    显示器
'''


def convert_from_color_segmentation(image_3d=None):
	import numpy as np
	from skimage import img_as_ubyte
	import warnings
	patterns = {
		(0, 0, 0):          0,      # 背景
		(128, 0, 0):        1,      # 飞机
		(0, 128, 0):        2,      # 自行车
		(128, 128, 0):      3,      # 鸟
		(0, 0, 128):        4,      # 船
		(128, 0, 128):      5,      # 瓶子
		(0, 128, 128):      6,      # 大巴
		(128, 128, 128):    7,      # 车
		(64, 0, 0):         8,      # 猫
		(192, 0, 0):        9,      # 椅子
		(64, 128, 0):       10,     # 牛
		(192, 128, 0):      11,     # 餐桌
		(64, 0, 128):       12,     # 狗
		(192, 0, 128):      13,     # 马
		(64, 128, 128):     14,     # 摩托车
		(192, 128, 128):    15,     # 人
		(0, 64, 0):         16,     # 盆栽
		(128, 64, 0):       17,     # 羊
		(0, 192, 0):        18,     # 沙发
		(128, 192, 0):      19,     # 火车
		(0, 64, 128):       20      # 显示器
	}
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		image_3d = img_as_ubyte(image_3d)
	image_2d = np.zeros((image_3d.shape[0], image_3d.shape[1]), dtype=np.uint8)
	for pattern, index in patterns.items():
		# print(pattern)
		# print(index)
		m = (image_3d == np.array(pattern).reshape(1, 1, 3)).all(axis=2)
		# print(m)
		image_2d[m] = index

	return image_2d


def write_img_to_tfrecords():
	import cv2
	# from skimage import io, transform
	import glob
	import numpy as np
	train_set_size = 10000
	development_set_size = 2000
	path = glob.glob(os.path.join('/home/dufanxin/PycharmProjects/ImagePreprocess/outputdata', '*.JPEG'))
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'development_set.tfrecords'))
	# test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件
	train_path = path[:10000]
	development_path = path[10000:]
	# print(len(path))

	for index, file_path in enumerate(train_path):
		train_image = cv2.imread(file_path)
		# train_image = cv2.resize(src=train_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		sample_image = np.asarray(a=train_image[:, :, 0], dtype=np.uint8)
		sample_image = cv2.resize(src=sample_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		label_image = np.asarray(a=train_image[:, :, 2], dtype=np.uint8)
		label_image = cv2.resize(src=label_image, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		label_image[label_image <= 100] = 0
		label_image[label_image > 100] = 1
		# train_image = io.imread(file_path)
		# train_image = transform.resize(train_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = train_image[:, :, 0]
		# label_image = train_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_image.tobytes()])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample_image.tobytes()]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done train_set writing %.2f%%' % (index / train_set_size * 100))
	train_set_writer.close()
	print("Done train_set writing")

	for index, file_path in enumerate(development_path):
		development_image = cv2.imread(file_path)
		# development_image = cv2.resize(src=development_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		sample_image = np.asarray(a=development_image[:, :, 0], dtype=np.uint8)
		sample_image = cv2.resize(src=sample_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		label_image = np.asarray(development_image[:, :, 2], dtype=np.uint8)
		label_image = cv2.resize(src=label_image, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		label_image[label_image <= 100] = 0
		label_image[label_image > 100] = 10
		# development_image = io.imread(file_path)
		# development_image = transform.resize(development_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = development_image[:, :, 0]
		# label_image = development_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_image.tobytes()])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample_image.tobytes()]))
		}))  # example对象对label和image数据进行封装
		development_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done development_set writing %.2f%%' % (index / development_set_size * 100))
	development_set_writer.close()
	print("Done development_set writing")


def read_check_tfrecords():
	import cv2
	train_file_path = os.path.join('../input-data/Segmentation', 'train_set.tfrecords')
	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	train_images, train_labels = read_image(train_image_filename_queue)
	one_hot_labels = tf.to_float(tf.one_hot(indices=train_labels, depth=CLASS_NUM))
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		example, label = sess.run([train_images, train_labels])
		example, label = sess.run([train_images, train_labels])
		cv2.imshow('image', example)
		cv2.imshow('lael', label * 100)
		cv2.waitKey(0)
		# print(sess.run(one_hot_labels))
		coord.request_stop()
		coord.join(threads)
	print("Done reading and checking")


def read_image(file_queue):
	reader = tf.TFRecordReader()
	# key, value = reader.read(file_queue)
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'label': tf.FixedLenFeature([], tf.string),
			'image_raw': tf.FixedLenFeature([], tf.string)
			})

	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# print('image ' + str(image))
	image = tf.reshape(image, [INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL])
	# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	# image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDE))
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	label = tf.decode_raw(features['label'], tf.uint8)
	# label = tf.cast(label, tf.int64)
	label = tf.reshape(label, [OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT])
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	min_after_dequeue = 2000
	capacity = 4000
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
	# one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
	return image_batch, one_hot_labels


class Unet:
	input_image = None
	input_label = None
	keep_prob = None
	lamb = None
	result_expand = None
	loss, loss_mean, loss_all, train_step = [None] * 4
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
			self.input_label = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE, CLASS_NUM], name='input_labels'
			)

			# for sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# not using one-hot coding
			# self.input_label = tf.placeholder(
			# 	dtype=tf.int32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
			# )
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
				output_shape=[batch_size, 20, 20, 512],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
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
			# print(result_relu_2.shape[1])

			# up sample
			self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_11')
			self.b[14] = self.init_b(shape=[256], name='b_11')
			result_up = tf.nn.conv2d_transpose(
				value=result_relu_2, filter=self.w[14],
				output_shape=[batch_size, 32, 32, 256],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
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
				output_shape=[batch_size, 56, 56, 128],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
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
				output_shape=[batch_size, 104, 104, 64],
				strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
			result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[20], name='add_bias'), name='relu_3')

		# layer 9
		with tf.name_scope('layer_9'):
			# copy, crop and merge
			result_merge = self.copy_and_crop_and_merge(
				result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_relu_3)

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

			# convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
			self.w[23] = self.init_w(shape=[1, 1, 64, CLASS_NUM], name='w_11')
			self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
			result_conv_3 = tf.nn.conv2d(
				input=result_relu_2, filter=self.w[23],
				strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
			self.prediction = tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias')
		# print(self.prediction)
		# print(self.input_label)

		# softmax loss
		with tf.name_scope('softmax_loss'):
			# using one-hot
			self.loss = \
				tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')

			# not using one-hot
			# self.loss = \
			# 	tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			self.loss_mean = tf.reduce_mean(self.loss)
			tf.add_to_collection(name='loss', value=self.loss_mean)
			self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

		# accuracy
		with tf.name_scope('accuracy'):
			# using one-hot
			self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.input_label, axis=3))

			# not using one-hot
			# self.correct_prediction = \
			# 	tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		# Gradient Descent
		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

	def train(self, train_files_queue):
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		train_images, train_labels = read_image_batch(train_files_queue, TRAIN_BATCH_SIZE)
		tf.summary.scalar("loss", self.loss_mean)
		tf.summary.scalar('accuracy', self.accuracy)
		merged_summary = tf.summary.merge_all()
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			# example, label = sess.run([train_images, train_labels])
			# result = sess.run(self.train_step, feed_dict={self.input_image: example, self.input_label: label})
			# print(result)
			# example, label = sess.run([train_images, train_labels])
			# lo, acc, summary_str = sess.run(
			# 	[self.loss_mean, self.accuracy, merged_summary],
			# 	feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 1.0, self.lamb: 0.004}
			# )
			# # summary_writer.add_summary(summary_str, index)
			# sess.run(
			# 	[self.train_step],
			# 	feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 0.6, self.lamb: 0.004}
			# )
			# print(example.shape)
			# print(label.shape)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					# print('epoch ' + str(epoch))
					example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# print(label)
					lo, acc, summary_str = sess.run(
						[self.loss_mean, self.accuracy, merged_summary],
						feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 1.0, self.lamb: 0.004}
					)
					summary_writer.add_summary(summary_str, epoch)
					# print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					if epoch % 10 == 0:
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
					sess.run(
						[self.train_step],
						feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 0.6, self.lamb: 0.004}
					)
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done train -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				all_parameters_saver.save(sess=sess, save_path=ckpt_path)
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print("Done training")


def main():
	# train
	train_file_path = os.path.join(FLAGS.data_dir, "train_set.tfrecords")
	# development
	# development_file_path = os.path.join(FLAGS.data_dir, "development_set.tfrecords")
	# test
	# test_file_path = os.path.join(FLAGS.data_dir, "test_set.tfrecords")
	# check point
	# ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
	# train_images, train_labels = read_image_batch(train_image_filename_queue, 80)

	# development_image_filename_queue = tf.train.string_input_producer(
	# 	tf.train.match_filenames_once(development_file_path), num_epochs=1, shuffle=True)
	# development_images, development_labels = read_image_batch(development_image_filename_queue, 5)

	# test_image_filename_queue = tf.train.string_input_producer(
	# 	tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=True)

	net = Unet()
	net.set_up_unet(TRAIN_BATCH_SIZE)
	net.train(train_image_filename_queue)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 数据地址
	parser.add_argument(
		'--data_dir', type=str, default='../input-data/Segmentation',
		help='Directory for storing input data')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='../output-data/models',
		help='output model path')

	# 模型保存地址
	parser.add_argument(
		'--tb_dir', type=str, default='../output-data/log',
		help='TensorBoard log path')

	FLAGS, _ = parser.parse_known_args()
	# write_img_to_tfrecords()
	# read_check_tfrecords()
	main()

