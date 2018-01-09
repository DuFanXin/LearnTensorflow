# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        LearnCNN 
# Date:             1/6/18 2:16 PM 
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
import sys
import argparse
# import matplotlib.pyplot as plt
from PIL import Image  # 注意Image,后面会用到
import Constant_Variables

IMG_WIDE = Constant_Variables.IMG_WIDE
IMG_HEIGHT = Constant_Variables.IMG_HEIGHT
IMG_CHANNEL = Constant_Variables.IMG_CHANNEL
EPOCH_NUM = Constant_Variables.EPOCH_NUM
LAYER_NUM = Constant_Variables.LAYER_NUM
BATCH_SIZE = Constant_Variables.BATCH_SIZE
BATCH_NUM = Constant_Variables.BATCH_NUM


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
	image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDE, IMG_CHANNEL])  # reshape为128*128的3通道图片
	# image.set_shape([IMG_HEIGH * IMG_WIDE * IMG_CHANNEL])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	# label = tf.decode_raw(features['image_raw'], tf.uint8)
	# print(label)
	# label = tf.reshape(label, shape=[1, 4])
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	capacity = 3 * batch_size
	image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=4))
	one_hot_labels = tf.reshape(label_batch, [batch_size, 1])
	return image_batch, one_hot_labels


def main():
	# train
	train_file_path = os.path.join(FLAGS.data_dir, "train_set.tfrecords")
	# development
	# development_file_path = os.path.join(FLAGS.data_dir, "development_set.tfrecords")
	# # test
	# test_file_path = os.path.join(FLAGS.data_dir, "test_set.tfrecords")
	# check point
	ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

	train_image_filename_queue = tf.train.string_input_producer(
		string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM)
	train_images, train_labels = read_image_batch(train_image_filename_queue, 4)

	# development_image_filename_queue = tf.train.string_input_producer(
	# 	tf.train.match_filenames_once(development_file_path))
	# development_images, development_labels = read_image_batch(development_image_filename_queue, 1)
	#
	# test_image_filename_queue = tf.train.string_input_producer(
	# 		tf.train.match_filenames_once(test_file_path))
	# test_images, test_labels = read_image_batch(test_image_filename_queue, 1)

	# layer1
	input_image = tf.placeholder(dtype=tf.float32, shape=[4, 128, 128, 3])
	output_label = tf.placeholder(dtype=tf.float32, shape=[4, 4])
	w_1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 3, 96]), name='w_1')
	b_1 = tf.Variable(initial_value=tf.random_normal(shape=[42, 42, 96]), name='b_1')
	result_1_conv = tf.nn.conv2d(
		input=input_image, filter=w_1,
		strides=[1, 3, 3, 1], padding='VALID')
	result_1_relu = tf.nn.relu(tf.add(result_1_conv, b_1), name='relu_1')
	result_1_maxpool = tf.nn.max_pool(
		value=result_1_relu, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1], padding='VALID')

	w_2 = tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 96, 256]), name='w_2')
	b_2 = tf.Variable(initial_value=tf.random_normal(shape=[7, 7, 256]), name='b_2')
	result_2_conv = tf.nn.conv2d(
		input=result_1_maxpool, filter=w_2,
		strides=[1, 3, 3, 1], padding='VALID')
	result_2_relu = tf.nn.relu(tf.add(result_2_conv, b_2), name='relu_2')
	result_2_maxpool = tf.nn.max_pool(
		value=result_2_relu, ksize=[1, 3, 3, 1],
		strides=[1, 2, 2, 1], padding='VALID')

	result_3_fc = tf.reshape(result_2_maxpool, [4, -1])

	w_4 = tf.Variable(tf.random_normal(shape=[2304, 100]), name='w_4')
	b_4 = tf.Variable(tf.random_normal(shape=[100]), name='b_4')
	result_4_fc = tf.matmul(result_3_fc, w_4) + b_4
	result_4_relu = tf.nn.relu(result_4_fc)

	w_5 = tf.Variable(tf.random_normal(shape=[100, 4]), name='w_5')
	b_5 = tf.Variable(tf.random_normal(shape=[4]), name='b_5')
	result_5_fc = tf.matmul(result_4_relu, w_5) + b_5

	result_6_softmax = tf.nn.softmax(result_5_fc)

	loss = tf.reduce_mean((output_label - result_6_softmax), 0)

	print(result_6_softmax)
	print(train_labels)
	# all_saver = tf.train.Saver()
	# with tf.Session() as sess:  # 开始一个会话
	# 	sess.run(tf.global_variables_initializer())
	# 	sess.run(tf.local_variables_initializer())
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(coord=coord)
	# 	for i in range(5):
	# 		example, lablel = sess.run([train_images, train_labels])  # 在会话中取出image和label
	# 		print(lablel)
	# 	# l = sess.run(w_1, feed_dict={input_image: example, output_label: a})
	#
	# 	# for i in range(Epoch_num):
	# 		# example, lablel = sess.run([train_images, train_labels])  # 在会话中取出image和label
	# 		# sess.run(result_conv_1, feed_dict={input_image_1: example})
	# 		# TODO minibatch
	# 		# image = Image.fromarray(example[2], 'RGB')  # 这里Image是之前提到的
	# 		# image.save('Label_' + str(lablel[2]) + '.jpg')  # 存下图片
	# 		# all_saver.save(sess, ckpt_path)
	# 	coord.request_stop()
	# 	coord.join(threads)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# tf.summary.FileWriter(FLAGS.data_dir, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			epoch = 1
			while not coord.should_stop():
				# Run training steps or whatever
				print('epoch' + str(epoch))
				epoch += 1
				for i in range(BATCH_NUM):
					example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					print(label)
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
		# coord.request_stop()
		coord.join(threads)
	print("Done compute")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 输入地址
	parser.add_argument(
		'--data_dir', type=str, default='/home/dufanxin/PycharmProjects/image/dog',
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
	# print(FLAGS.data_dir)
	# tf.app.run(main=main)
	# write_img_to_tfrecords()
	# read_img_from_tfrecord_and_save()
	main()
