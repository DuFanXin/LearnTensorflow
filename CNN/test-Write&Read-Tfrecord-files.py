# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        test-Write&Read-Tfrecord-files 
# Date:             1/10/18 9:33 PM 
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
import glob

IMG_WIDE = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3


def main():
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'development_set.tfrecords'))
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	files_path = glob.glob(os.path.join(FLAGS.data_dir, '*.jpg'))
	# files_path = os.path.join(FLAGS.data_dir, 'n02512053_4.jpg')
	# img = Image.open(files_path)
	# img = img.resize((IMG_HEIGHT, IMG_WIDE))
	# img_raw = img.tobytes()  # 将图片转化为二进制格式
	# # 为了使用softmax函数，需要给出类似[0, 1, 0, ..., 0]的label
	# example = tf.train.Example(features=tf.train.Features(feature={
	# 	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[4])),
	# 	'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	# }))  # example对象对label和image数据进行封装
	# train_set_writer.write(example.SerializeToString())  # 序列化为字符串
	#
	# # print(index % 4)
	# train_set_writer.close()
	# print('Done train_set write')
	print('files num ' + str(len(files_path)))
	length = len(files_path)
	development_set_size, test_set_size = length >= 100 and (length//100, length//100) or (1, 1)
	train_set_size = length - development_set_size - test_set_size
	# train_set_size = development_set_size = test_set_size = 4
	train_set_path = files_path[:train_set_size]
	development_set_path = files_path[train_set_size:train_set_size + development_set_size]
	test_set_path = files_path[train_set_size + development_set_size:]
	print('train files num ' + str(len(train_set_path)))
	print('train files num ' + str(len(development_set_path)))
	print('test files num ' + str(len(test_set_path)))
	for index, image_path in enumerate(train_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_HEIGHT, IMG_WIDE))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		# 为了使用softmax函数，需要给出类似[0, 1, 0, ..., 0]的label
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index % 4])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index % 4)
	train_set_writer.close()
	print('Done train_set write')

	for index, image_path in enumerate(development_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_HEIGHT, IMG_WIDE))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index % 4])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		development_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index % 4)
	development_set_writer.close()
	print('Done development_set write')

	for index, image_path in enumerate(test_set_path):
		img = Image.open(image_path)
		img = img.resize((IMG_HEIGHT, IMG_WIDE))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index % 4])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index % 4)
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
	image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDE, IMG_CHANNEL])  # reshape为128*128的3通道图片
	# image.set_shape([IMG_HEIGH * IMG_WIDE * IMG_CHANNEL])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	min_after_dequeue = 2 * batch_size
	capacity = 3 * batch_size + min_after_dequeue
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
			tensors=[img, label], batch_size=batch_size,
			capacity=capacity, min_after_dequeue=min_after_dequeue
	)
	# one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
	one_hot_labels = tf.reshape(label_batch, [batch_size, 1])
	return image_batch, one_hot_labels


def check_proprocess():
	# train
	train_file_path = os.path.join(FLAGS.data_dir, "train_set.tfrecords")
	# development
	development_file_path = os.path.join(FLAGS.data_dir, "development_set.tfrecords")
	# test
	test_file_path = os.path.join(FLAGS.data_dir, "test_set.tfrecords")
	#
	train_image_filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(train_file_path), num_epochs=1)
	train_images, train_labels = read_image_batch(train_image_filename_queue, 49)
	#
	development_image_filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(development_file_path))
	development_images, development_labels = read_image_batch(development_image_filename_queue, 1)

	test_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(test_file_path))
	test_images, test_labels = read_image_batch(test_image_filename_queue, 1)
	#
	# print(train_images)
	# print(train_labels)

	# with tf.Session() as sess:  # 开始一个会话
	# 	sess.run(tf.global_variables_initializer())
	# 	sess.run(tf.local_variables_initializer())
	# 	# print(sess.run(a))
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(coord=coord)
	# 	# for i in range(5):
	# 	example, lablel = sess.run([train_images, train_labels])  # 在会话中取出image和label
	# 		# DO check
	# 		# image = Image.fromarray(example[2], 'RGB')  # 这里Image是之前提到的
	# 		# image.save('Label_' + str(lablel[2]) + '.jpg')  # 存下图片
	# 	coord.request_stop()
	# 	coord.join(threads)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			epoch = 1
			while not coord.should_stop():
				# Run training steps or whatever
				print('epoch' + str(epoch))
				epoch += 1
				example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
				# print(label)
		except tf.errors.OutOfRangeError:
			print('Done -- epoch limit reached')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
		# coord.request_stop()
		coord.join(threads)
	print("check done")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 输入地址
	parser.add_argument(
		'--data_dir', type=str, default='/home/dufanxin/PycharmProjects/image/fish',
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
	main()
	check_proprocess()
