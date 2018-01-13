# -*- coding:utf-8 -*-
'''
#====#====#====#====
# Project Name:     LearnTensorflow
# File Name:        FishOrDog
# Date:             1/8/18 2:33 PM
# Using IDE:        PyCharm Community Edition
# From HomePage:    https://github.com/DuFanXin/LearnTensorflow
# Author:           DuFanXin
# BlogPage:         http://blog.csdn.net/qq_30239975
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#====
'''

import tensorflow as tf
import os
import argparse
# import matplotlib.pyplot as plt
from PIL import Image  # 注意Image,后面会用到
import glob

IMG_WIDE, IMG_HEIGHT, IMG_CHANNEL = 227, 227, 3
EPOCH_NUM = 50
TRAIN_BATCH_SIZE = 80


class Net:
	input_image, input_label = [None] * 2
	w_1, w_2, w_3, w_4, w_5, w_6, w_7 = [None] * 7
	result_1_conv, result_2_conv, result_3_conv, result_4_conv = [None] * 4
	b_1, b_2, b_3, b_4, b_5, b_6, b_7 = [None] * 7
	result_1_relu, result_2_relu, result_3_relu, result_4_relu, result_5_relu, result_6_relu, result_7_relu = [None] * 7
	result_5_fc, result_6_fc, result_7_fc = [None] * 3
	result_1_maxpool, result_2_maxpool, result_3_maxpool, result_4_maxpool = [None] * 4
	result_expand = None
	result_8_softmax = None
	loss, loss_mean = [None] * 2
	train_step = None

	def __init__(self):
		print('new network')

	def set_up_network(self):
		# input and output
		self.input_image = tf.placeholder(dtype=tf.float32, shape=[TRAIN_BATCH_SIZE, IMG_WIDE, IMG_WIDE, IMG_CHANNEL])
		self.input_label = tf.placeholder(dtype=tf.float32, shape=[TRAIN_BATCH_SIZE, 2])

		# layer 1
		self.w_1 = tf.Variable(initial_value=tf.random_normal(shape=[11, 11, 3, 96], dtype=tf.float32), name='w_1')
		self.b_1 = tf.Variable(initial_value=tf.random_normal(shape=[55, 55, 96], dtype=tf.float32), name='b_1')
		self.result_1_conv = tf.nn.conv2d(
			input=self.input_image, filter=self.w_1,
			strides=[1, 4, 4, 1], padding='VALID')
		self.result_1_relu = tf.nn.relu(tf.add(self.result_1_conv, self.b_1), name='relu_1')
		self.result_1_maxpool = tf.nn.max_pool(
			value=self.result_1_relu, ksize=[1, 3, 3, 1],
			strides=[1, 2, 2, 1], padding='VALID')

		# layer 2
		self.w_2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 96, 256], dtype=tf.float32), name='w_2')
		self.b_2 = tf.Variable(initial_value=tf.random_normal(shape=[27, 27, 256], dtype=tf.float32), name='b_2')
		self.result_2_conv = tf.nn.conv2d(
			input=self.result_1_maxpool, filter=self.w_2,
			strides=[1, 1, 1, 1], padding='SAME')
		self.result_2_relu = tf.nn.relu(tf.add(self.result_2_conv, self.b_2), name='relu_2')
		self.result_2_maxpool = tf.nn.max_pool(
			value=self.result_2_relu, ksize=[1, 3, 3, 1],
			strides=[1, 2, 2, 1], padding='VALID')

		# layer 3
		self.w_3 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 256, 384], dtype=tf.float32), name='w_3')
		self.b_3 = tf.Variable(initial_value=tf.random_normal(shape=[13, 13, 384], dtype=tf.float32), name='b_3')
		self.result_3_conv = tf.nn.conv2d(
			input=self.result_2_maxpool, filter=self.w_3,
			strides=[1, 1, 1, 1], padding='SAME')
		self.result_3_relu = tf.nn.relu(tf.add(self.result_3_conv, self.b_3), name='relu_3')
		self.result_3_maxpool = tf.nn.max_pool(
			value=self.result_3_relu, ksize=[1, 3, 3, 1],
			strides=[1, 1, 1, 1], padding='SAME')

		# layer 4
		self.w_4 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 384, 256], dtype=tf.float32), name='w_4')
		self.b_4 = tf.Variable(initial_value=tf.random_normal(shape=[13, 13, 256], dtype=tf.float32), name='b_4')
		self.result_4_conv = tf.nn.conv2d(
			input=self.result_3_maxpool, filter=self.w_4,
			strides=[1, 1, 1, 1], padding='SAME')
		self.result_4_relu = tf.nn.relu(tf.add(self.result_4_conv, self.b_4), name='relu_4')
		self.result_4_maxpool = tf.nn.max_pool(
			value=self.result_4_relu, ksize=[1, 3, 3, 1],
			strides=[1, 2, 2, 1], padding='VALID')

		# expand to [TRAIN_BATCH_SIZE, -1]
		self.result_expand = tf.reshape(self.result_4_maxpool, [TRAIN_BATCH_SIZE, -1])

		# layer 5
		self.w_5 = tf.Variable(initial_value=tf.random_normal(shape=[9216, 4096], dtype=tf.float32), name='w_5')
		self.b_5 = tf.Variable(initial_value=tf.random_normal(shape=[4096], dtype=tf.float32), name='b_5')
		self.result_5_fc = tf.add(tf.matmul(self.result_expand, self.w_5), self.b_5, name='result_5_fc')
		self.result_5_relu = tf.nn.relu(self.result_5_fc, name='result_5_relu')

		# layer 6
		self.w_6 = tf.Variable(initial_value=tf.random_normal(shape=[4096, 1024], dtype=tf.float32), name='w_6')
		self.b_6 = tf.Variable(initial_value=tf.random_normal(shape=[1024], dtype=tf.float32), name='b_6')
		self.result_6_fc = tf.add(tf.matmul(self.result_5_relu, self.w_6), self.b_6, name='result_6_fc')
		self.result_6_relu = tf.nn.relu(self.result_6_fc, name='result_6_relu')

		# layer 7
		self.w_7 = tf.Variable(initial_value=tf.random_normal(shape=[1024, 2], dtype=tf.float32), name='w_7')
		self.b_7 = tf.Variable(initial_value=tf.random_normal(shape=[2], dtype=tf.float32), name='b_7')
		self.result_7_fc = tf.add(tf.matmul(self.result_6_relu, self.w_7), self.b_7, name='result_7_fc')
		self.result_7_relu = tf.nn.relu(self.result_7_fc, name='result_7_relu')

		# layer 8
		self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.result_7_relu)
		self.loss_mean = tf.reduce_mean(self.loss)

		# Gradient Descent
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss_mean)

	def train(self):
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss_mean)


def change_prefix(files):   # 用来吧JPEG转换为JPG
	for filename in files:
		portion = os.path.splitext(filename)  # 分离文件名字和后缀
		# print(portion)
		if portion[1] == ".JPEG":  # 根据后缀来修改,如无后缀则空
			newname = portion[0] + ".jpg"  # 要改的新后缀
			os.rename(filename, newname)


def write_img_to_tfrecords():

	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'development_set.tfrecords'))
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	# fish_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'fish/*.JPEG'))
	# dog_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'dog/*.JPEG'))
	# change_prefix(fish_image_path)
	# change_prefix(dog_image_path)
	fish_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'fish/*.jpg'))
	dog_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'dog/*.jpg'))
	label = {'dog': 1, 'fish': 2}
	# print(len(dog_image_path))
	# print('files num ' + str(len(files_path)))
	# dog_image_path_length = len(dog_image_path)
	# fish_image_path_length = len(fish_image_path)
	#
	# dog_development_set_size, dog_test_set_size = \
	# 	dog_image_path_length >= 100 and (dog_image_path_length//100, dog_image_path_length//100) or (1, 1)
	# fish_development_set_size, fish_test_set_size = \
	# 	fish_image_path_length >= 100 and (fish_image_path_length // 100, fish_image_path_length // 100) or (1, 1)
	# dog_train_set_size = dog_image_path_length - dog_development_set_size - dog_test_set_size
	# fish_train_set_size = fish_image_path_length - fish_development_set_size - fish_test_set_size
	#
	# # train_set_size = development_set_size = test_set_size = 4
	# train_set_path = dog_image_path[:dog_train_set_size] + fish_image_path[:fish_train_set_size]
	# development_set_path = \
	# 	dog_image_path[dog_train_set_size:dog_train_set_size + dog_development_set_size] \
	# 	+ fish_image_path[fish_train_set_size:fish_train_set_size + fish_development_set_size]
	# test_set_path = \
	# 	dog_image_path[dog_train_set_size + dog_development_set_size:] \
	# 	+ fish_image_path[fish_train_set_size + fish_development_set_size:]
	# print(len(test_set_path))

	'''dog images'''
	dog_length = len(dog_image_path)
	dog_development_set_size, dog_test_set_size = \
		dog_length >= 100 and (dog_length // 100, dog_length // 100) or (1, 1)
	dog_train_set_size = dog_length - dog_development_set_size - dog_test_set_size
	# train_set_size = development_set_size = test_set_size = 4
	train_set_path = dog_image_path[:dog_train_set_size]
	development_set_path = dog_image_path[dog_train_set_size:dog_train_set_size + dog_development_set_size]
	test_set_path = dog_image_path[dog_train_set_size + dog_development_set_size:]

	# print('train files num ' + str(len(train_set_path)))
	# print('train files num ' + str(len(development_set_path)))
	# print('test files num ' + str(len(test_set_path)))
	for index, image_path in enumerate(train_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_WIDE, IMG_HEIGHT))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['dog']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	# train_set_writer.close()
	# print('Done train_set write')

	for index, image_path in enumerate(development_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_WIDE, IMG_HEIGHT))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['dog']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		development_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	# development_set_writer.close()
	# print('Done development_set write')

	for index, image_path in enumerate(test_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_WIDE, IMG_HEIGHT))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['dog']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	# test_set_writer.close()
	# print('Done test_set write')

	'''fish images'''
	fish_length = len(fish_image_path)
	fish_development_set_size, fish_test_set_size = \
		fish_length >= 100 and (fish_length // 100, fish_length // 100) or (1, 1)
	fish_train_set_size = fish_length - fish_development_set_size - fish_test_set_size
	# train_set_size = development_set_size = test_set_size = 4
	train_set_path = fish_image_path[:fish_train_set_size]
	development_set_path = fish_image_path[fish_train_set_size:fish_train_set_size + fish_development_set_size]
	test_set_path = fish_image_path[fish_train_set_size + fish_development_set_size:]

	print('train files num ' + str(dog_train_set_size + fish_train_set_size))
	print('development files num ' + str(dog_development_set_size + fish_development_set_size))
	print('test files num ' + str(dog_test_set_size + fish_test_set_size))
	for index, image_path in enumerate(train_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_WIDE, IMG_HEIGHT))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['fish']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	train_set_writer.close()
	print('Done train_set write')

	for index, image_path in enumerate(development_set_path):
		img = Image.open(image_path)
		img = img.resize((227, 227))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['fish']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	development_set_writer.close()
	print('Done development_set write')

	for index, image_path in enumerate(test_set_path):
		img = Image.open(image_path)
		img = img.resize((227, 227))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['fish']])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	test_set_writer.close()
	print('Done test_set write')


def read_image(file_queue):
	reader = tf.TFRecordReader()
	# key, value = reader.read(file_queue)
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
			})

	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# print('image ' + str(image))
	image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDE, IMG_CHANNEL])
	# image.set_shape([IMG_HEIGH * IMG_WIDE * IMG_CHANNEL])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	min_after_dequeue = 2 * batch_size
	capacity = 3 * batch_size + min_after_dequeue
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=2))
	# one_hot_labels = tf.reshape(label_batch, [batch_size, 1])
	return image_batch, one_hot_labels


def main():
	# train
	train_file_path = os.path.join(FLAGS.data_dir, "train_set.tfrecords")
	# development
	development_file_path = os.path.join(FLAGS.data_dir, "development_set.tfrecords")
	# test
	test_file_path = os.path.join(FLAGS.data_dir, "test_set.tfrecords")
	# check point
	ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
	train_images, train_labels = read_image_batch(train_image_filename_queue, 80)

	development_image_filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(development_file_path), num_epochs=EPOCH_NUM, shuffle=True)
	development_images, development_labels = read_image_batch(development_image_filename_queue, 5)

	test_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(test_file_path), num_epochs=EPOCH_NUM, shuffle=True)
	test_images, test_labels = read_image_batch(test_image_filename_queue, 5)

	# net
	net = Net()
	net.set_up_network()

	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# tf.summary.FileWriter(FLAGS.data_dir, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		example, label = sess.run([train_images, train_labels])
		result = sess.run(net.train_step, feed_dict={net.input_image: example, net.input_label: label})
		print(result)
		# example, label = sess.run([train_images, train_labels])
		# print(label)
		# try:
		# 	epoch = 1
		# 	while not coord.should_stop():
		# 		# Run training steps or whatever
		# 		print('epoch' + str(epoch))
		# 		# example, label = sess.run([test_images, test_labels])  # 在会话中取出image和label
		# 		# print(label)
		# 		for i in range(5):
		# 			example, label = sess.run([test_images, test_labels])  # 在会话中取出image和label
		# 			print(label)
		# 		epoch += 1
		# except tf.errors.OutOfRangeError:
		# 	print('Done -- epoch limit reached')
		# finally:
		# 	# When done, ask the threads to stop.
		# 	coord.request_stop()
		coord.request_stop()
		coord.join(threads)
	print("Done compute")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 输入地址
	parser.add_argument(
		'--data_dir', type=str, default='/home/dufanxin/PycharmProjects/image',
		help='input data path')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='',
		help='output model path')

	# 日志地址
	parser.add_argument(
		'--output_dir', type=str, default='',
		help='output data path')
	FLAGS, _ = parser.parse_known_args()
	# write_img_to_tfrecords()
	main()
