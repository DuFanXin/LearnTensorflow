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
EPOCH_NUM = 1


def change_prefix(files):
	for filename in files:
		portion = os.path.splitext(filename)  # 分离文件名字和后缀
		# print(portion)
		if portion[1] == ".JPEG":  # 根据后缀来修改,如无后缀则空
			newname = portion[0] + ".jpg"  # 要改的新后缀
			os.rename(filename, newname)


def write_img_to_tfrecords():

	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'development_set.tfrecords'))  # 要生成的文件
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	# fish_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'fish/*.JPEG'))
	dog_image_path = glob.glob(os.path.join(FLAGS.data_dir, 'dog/*.JPEG'))
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

	print('train files num ' + str(len(train_set_path)))
	print('train files num ' + str(len(development_set_path)))
	print('test files num ' + str(len(test_set_path)))
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
	train_set_writer.close()
	print('Done train_set write')

	for index, image_path in enumerate(development_set_path):
		img = Image.open(image_path)
		img = img.resize((227, 227))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		development_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	development_set_writer.close()
	print('Done development_set write')

	for index, image_path in enumerate(test_set_path):
		img = Image.open(image_path)
		img = img.resize((227, 227))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# print(index)
	test_set_writer.close()
	print('Done test_set write')

	# '''fish images'''
	# fish_length = len(fish_image_path)
	# fish_development_set_size, fish_test_set_size = \
	# 	fish_length >= 100 and (fish_length // 100, fish_length // 100) or (1, 1)
	# fish_train_set_size = fish_length - fish_development_set_size - fish_test_set_size
	# # train_set_size = development_set_size = test_set_size = 4
	# train_set_path = fish_image_path[:fish_train_set_size]
	# development_set_path = fish_image_path[fish_train_set_size:fish_train_set_size + fish_development_set_size]
	# test_set_path = fish_image_path[fish_train_set_size + fish_development_set_size:]
	#
	# print('train files num ' + str(dog_train_set_size + fish_train_set_size))
	# print('development files num ' + str(dog_development_set_size + fish_development_set_size))
	# print('test files num ' + str(dog_test_set_size + fish_test_set_size))
	# for index, image_path in enumerate(train_set_path):
	# 	img = Image.open(image_path)
	# 	img = img.resize((IMG_WIDE, IMG_HEIGHT))
	# 	img_raw = img.tobytes()  # 将图片转化为二进制格式
	# 	example = tf.train.Example(features=tf.train.Features(feature={
	# 		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label['fish']])),
	# 		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	# 	}))  # example对象对label和image数据进行封装
	# 	train_set_writer.write(example.SerializeToString())  # 序列化为字符串
	# 	# print(index)
	# train_set_writer.close()
	# print('Done train_set write')
	#
	# for index, image_path in enumerate(development_set_path):
	# 	img = Image.open(image_path)
	# 	img = img.resize((227, 227))
	# 	img_raw = img.tobytes()  # 将图片转化为二进制格式
	# 	example = tf.train.Example(features=tf.train.Features(feature={
	# 		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
	# 		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	# 	}))  # example对象对label和image数据进行封装
	# 	train_set_writer.write(example.SerializeToString())  # 序列化为字符串
	# 	# print(index)
	# development_set_writer.close()
	# print('Done development_set write')
	#
	# for index, image_path in enumerate(test_set_path):
	# 	img = Image.open(image_path)
	# 	img = img.resize((227, 227))
	# 	img_raw = img.tobytes()  # 将图片转化为二进制格式
	# 	example = tf.train.Example(features=tf.train.Features(feature={
	# 		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
	# 		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	# 	}))  # example对象对label和image数据进行封装
	# 	test_set_writer.write(example.SerializeToString())  # 序列化为字符串
	# 	# print(index)
	# test_set_writer.close()
	# print('Done test_set write')


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
	image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	# image_batch, label_batch = tf.train.shuffle_batch(
	# 	tensors=[img, label], batch_size=batch_size,
	# 	capacity=capacity, min_after_dequeue=min_after_dequeue)
	# one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=2))
	one_hot_labels = tf.reshape(label_batch, [batch_size, 1])
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

	# development_image_filename_queue = tf.train.string_input_producer(
	# 	tf.train.match_filenames_once(development_file_path), num_epochs=EPOCH_NUM)
	# development_images, development_labels = read_image_batch(development_image_filename_queue, 1)
	#
	# test_image_filename_queue = tf.train.string_input_producer(
	# 		tf.train.match_filenames_once(test_file_path), num_epochs=EPOCH_NUM)
	# test_images, test_labels = read_image_batch(test_image_filename_queue, 1)

	# layer1
	input_image = tf.placeholder(dtype=tf.float32, shape=[4, 128, 128, 3])
	output_label = tf.placeholder(dtype=tf.float32, shape=[4, 4])

	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# tf.summary.FileWriter(FLAGS.data_dir, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		example, label = sess.run([train_images, train_labels])
		# example, label = sess.run([train_images, train_labels])
		print(label)
		# try:
		# 	epoch = 1
		# 	while not coord.should_stop():
		# 		# Run training steps or whatever
		# 		print('epoch' + str(epoch))
		# 		epoch += 1
		# 		for i in range(BATCH_NUM):
		# 			example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
		# 			print(label)
		# except tf.errors.OutOfRangeError:
		# 	print('Done training -- epoch limit reached')
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
	# write_img_to_tfrecords()
	main()
