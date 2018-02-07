# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        TestSomeFuctions 
# Date:             1/15/18 10:01 AM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/LearnTensorflow
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import tensorflow as tf
from PIL import Image
import cv2
import glob
import os
import numpy as np
from skimage import io, transform, img_as_ubyte
import warnings
# import matplotlib.pyplot as plt
# img = Image.open('../input-data/MNIST/pictures/trainimage/pic2/0/3113.bmp')
# fig = plt.figure()
# plt.imshow(img, cmap = 'binary')#黑白显示
# plt.show()
# img.show()
CLASS_NUM, TRAIN_BATCH_SIZE = 2, 1
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 284, 284, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 100, 100, 1


def convert_from_color_segmentation(image_3d=None):
	# import numpy as np
	patterns = {
		(0, 0, 0): 0,
		(128, 0, 0): 1,
		(0, 128, 0): 2,
		(128, 128, 0): 3,
		(0, 0, 128): 4,
		(128, 0, 128): 5,
		(0, 128, 128): 6,
		(128, 128, 128): 7,
		(64, 0, 0): 8,
		(192, 0, 0): 9,
		(64, 128, 0): 10,
		(192, 128, 0): 11,
		(64, 0, 128): 12,
		(192, 0, 128): 13,
		(64, 128, 128): 14,
		(192, 128, 128): 15,
		(0, 64, 0): 16,
		(128, 64, 0): 17,
		(0, 192, 0): 18,
		(128, 192, 0): 19,
		(0, 64, 128): 20
	}
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		image_3d = img_as_ubyte(image_3d)
	image_2d = np.zeros((image_3d.shape[0], image_3d.shape[1]), dtype=np.uint8)
	for pattern, index in patterns.items():
		# print(pattern)
		# print(index)
		match = (image_3d == np.array(pattern).reshape(1, 1, 3)).all(axis=2)
		# print(m)
		image_2d[match] = index

	return image_2d

arr = (1, 1, 1)
Map = {arr: 3}
img = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.int32, shape=[28, 28, 3]))
a = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[5, 28, 28, 1024]))

w = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[2, 2, 512, 1024]))
b = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[512]))
result_up = tf.nn.conv2d_transpose(
				value=a, filter=w,
				output_shape=[5, 56, 56, 512], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
result_conv = tf.nn.relu(tf.nn.bias_add(result_up, b, name='add_bias'), name='relu_3')

file_name = b'2007_000063'


def write():
	train_set_writer = tf.python_io.TFRecordWriter(
		os.path.join('../input-data/Segmentation', 'train_set_temp.tfrecords'))
	path = os.path.join('../input-data/Segmentation', 'ImageSets/train.txt')
	file_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(path), num_epochs=1, shuffle=True)
	reader = tf.TextLineReader()
	key, value = reader.read(file_queue)
	defaults = [['string']]
	files_name = tf.decode_csv(records=value, record_defaults=defaults)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		file_name = sess.run(files_name)
		print(file_name[0])
		# print(os.path.join('../input-data/Segmentation', 'TrainImage/%s.jpg' % str(file_name[0], encoding='utf-8')))
		trainImage_value = tf.gfile.FastGFile(
			os.path.join('../input-data/Segmentation', 'TrainImage/%s.jpg' % str(file_name[0], encoding='utf-8')),
			'rb').read()
		trainImage_tensor = tf.image.decode_jpeg(trainImage_value, channels=3)
		trainImage_resized = tf.image.resize_images(trainImage_tensor, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		trainImage_arry = sess.run(trainImage_resized)
		print(trainImage_arry.shape)
		trainImage_string = trainImage_arry.tostring()

		labelImage_value = tf.gfile.FastGFile(
			os.path.join('../input-data/Segmentation', 'LabelImage/%s.png' % str(file_name[0], encoding='utf-8')),
			'rb').read()
		labelImage_tensor = tf.image.decode_png(labelImage_value, channels=3)
		labelImage_resized = tf.image.resize_images(labelImage_tensor, (OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE))
		labelImage_arry = sess.run(labelImage_resized)
		# labelImage_arry = tf.cast(labelImage_arry, tf.int32)
		labelImage_converted = convert_from_color_segmentation(labelImage_arry)
		# print(labelImage_arry)
		# print(labelImage_converted)
		labelImage_string = labelImage_converted.tostring()

		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labelImage_string])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[trainImage_string]))
		}))
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		coord.request_stop()
		coord.join(threads)
		train_set_writer.close()
	print("Done write")


def write_2():
	train_set_writer = tf.python_io.TFRecordWriter(
		os.path.join('../input-data/Segmentation', 'train_set_temp.tfrecords'))
	train_img = Image.open(
		os.path.join('../input-data/Segmentation', 'TrainImage/%s.jpg' % str(file_name, encoding='utf-8')))
	train_img = train_img.resize((INPUT_IMG_HEIGHT, INPUT_IMG_WIDE))
	# train_img_raw = train_img.tobytes()  # 将图片转化为二进制格式

	# train_img = io.imread(
	# 	os.path.join('../input-data/Segmentation', 'TrainImage/%s.jpg' % str(file_name, encoding='utf-8')))
	# train_img = transform.resize(train_img, (INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, 3))

	# label_img = Image.open(
	# 	os.path.join('../input-data/Segmentation', 'LabelImage/%s.png' % str(b'2007_000129', encoding='utf-8')))
	# label_img = label_img.resize((INPUT_IMG_HEIGHT, INPUT_IMG_WIDE))
	# # print(label_img)
	# label_img_raw = label_img.tobytes()  # 将图片转化为二进制格式

	label_img = io.imread(
		os.path.join('../input-data/Segmentation', 'LabelImage/%s.png' % str(file_name, encoding='utf-8')))
	# io.imshow(data.astronaut())
	label_img = transform.resize(label_img, (OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, 3))
	# print(label_img.dtype)
	# print(label_img)
	# label_img = img_as_ubyte(label_img)
	# print(label_img)
	label_img = convert_from_color_segmentation(label_img) * 10
	# label_img = transform.resize(label_img, (OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, 1))
	io.imsave(
		fname=os.path.join('../input-data/Segmentation', 'Labels/%s.jpg' % str(file_name, encoding='utf-8')), arr=label_img)
	print(label_img)
	# label.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_img.tostring()])),
		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_img.tobytes()]))
	}))  # example对象对label和image数据进行封装
	train_set_writer.write(example.SerializeToString())  # 序列化为字符串
	# print(index % 4)
	train_set_writer.close()


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
			print(self.input_image)
			# for softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# using one-hot
			# self.input_label = tf.placeholder(
			# 	dtype=tf.float32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE, CLASS_NUM], name='input_labels'
			# )

			# for sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
			# not using one-hot coding
			self.input_label = tf.placeholder(
				dtype=tf.int32, shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
			)
			print(self.input_label)
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
			self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

		# accuracy
		with tf.name_scope('accuracy'):
			# using one-hot
			# self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_label, 1))

			# not using one-hot
			self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)

		# Gradient Descent
		with tf.name_scope('Gradient_Descent'):
			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

		print('U-net setted')

	def train(self, train_files_queue):
		# ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		train_images, train_labels = read_image_batch(train_files_queue, TRAIN_BATCH_SIZE)
		# all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  # 开始一个会话
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			# summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			# example, label = sess.run([train_images, train_labels])
			# result = sess.run(self.train_step, feed_dict={self.input_image: example, self.input_label: label})
			# print(result)
			example, label = sess.run([train_images, train_labels])
			print(example.shape)
			print(label.shape)
			lo, acc = sess.run(
				[self.loss_all, self.accuracy],
				feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 1.0, self.lamb: 0.004}
			)
			# summary_writer.add_summary(summary_str, index)
			print('loss: %.6f and accuracy: %.6f' % (lo, acc))
			sess.run(
				[self.train_step],
				feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 0.6, self.lamb: 0.004}
			)
			coord.request_stop()
			coord.join(threads)
		print("Done training")


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
	# print('image ' + str(type(image)))
	# image = tf.reshape(image, [INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, 3])
	image = tf.reshape(image, [INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL])
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	label = tf.decode_raw(features['label'], tf.uint8)
	# label = tf.cast(features['label'], tf.int64)
	label = tf.reshape(label, [OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
	# label = tf.to_float(tf.one_hot(indices=label, depth=CLASS_NUM))
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	# print(img)
	# print(label)
	min_after_dequeue = 2000
	capacity = 4000
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	# image_batch, label_batch = img, label
	label_batch = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
	# one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
	return image_batch, label_batch


def read():
	train_file_path = os.path.join('../input-data/Segmentation', 'train_set.tfrecords')

	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		example, label = sess.run([train_images, train_labels])
		coord.request_stop()
		coord.join(threads)
	print("Done reading")


def test_softmax():
	import cv2
	# y =
	# tf.Variable(initial_value=tf.random_normal(shape=[OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM], dtype=tf.float32))
	y = tf.constant(value=1, shape=[TRAIN_BATCH_SIZE, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT], dtype=tf.uint8)
	y = tf.to_float(tf.one_hot(indices=y, depth=CLASS_NUM))
	train_file_path = os.path.join('../input-data/Segmentation', 'train_set.tfrecords')
	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	# train_images, train_labels = read_image(train_image_filename_queue)
	train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		example, label = sess.run([train_images, train_labels])
		print(label.shape)
		correct_prediction = tf.equal(tf.argmax(y, dimension=3), tf.argmax(label, dimension=3))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		cp_img = sess.run(tf.cast(correct_prediction, tf.uint8))[0] * 100
		accuracy = tf.reduce_mean(correct_prediction)
		reshape_pred = sess.run(tf.cast(x=tf.argmax(y, dimension=3), dtype=tf.uint8))[0] * 100
		reshape_label = sess.run(tf.cast(x=tf.argmax(label, dimension=3), dtype=tf.uint8))[0] * 100
		# cv2.imshow('img', example)
		# cv2.imshow('label', sess.run(tf.argmax(label, axis=2) * 100))
		cv2.imshow('label', reshape_label)
		cv2.imshow('pred', reshape_pred)
		cv2.imshow('cp_img', cp_img)
		cv2.waitKey(0)
		print(cp_img.shape)
		print(sess.run(accuracy))
		# print(sess.run(tf.argmax(label, axis=2)))
		coord.request_stop()
		coord.join(threads)
	print("Done reading")


if __name__ == '__main__':
	# image = cv2.imread(os.path.join('../input-data/Segmentation', 'LabelImage/%s.png' % str(file_name, encoding='utf-8')))
	# cv2.imshow('image', image)
	test_softmax()
	# write()
	# write_2()
	# read()
	# net = Unet()
	# net.set_up_unet(TRAIN_BATCH_SIZE)
	#
	# train_file_path = os.path.join('../input-data/Segmentation', 'train_set_temp.tfrecords')
	# train_image_filename_queue = tf.train.string_input_producer(
	# 	string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
	# net.train(train_image_filename_queue)
