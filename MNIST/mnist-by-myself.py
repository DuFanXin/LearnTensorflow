# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     LearnTensorflow 
# File Name:        mnist-by-myself 
# Date:             1/16/18 8:22 PM 
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

IMG_WIDE, IMG_HEIGHT, IMG_CHANNEL = 32, 32, 1
EPOCH_NUM = 10
TRAIN_BATCH_SIZE = 128
DEVELOPMENT_BATCH_SIZE = 10
TEST_BATCH_SIZE = 128
EPS = 10e-5
FLAGS = None
CLASS_NUM = 10


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

	def __init__(self):
		print('new network')

	def set_up_network(self, batch_size):
		# input and output
		with tf.name_scope('input'):
			self.input_image = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, IMG_WIDE, IMG_WIDE, IMG_CHANNEL], name='input_images'
			)
			self.input_label = tf.placeholder(
				dtype=tf.float32, shape=[batch_size, CLASS_NUM], name='input_labels'
			)
			self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

		# layer 1
		with tf.name_scope('layer_1'):
			self.w_1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, IMG_CHANNEL, 6], dtype=tf.float32), name='w_1')
			self.b_1 = tf.Variable(initial_value=tf.random_normal(shape=[6], dtype=tf.float32), name='b_1')
			self.result_1_conv = tf.nn.conv2d(
				input=self.input_image, filter=self.w_1,
				strides=[1, 1, 1, 1], padding='VALID', name='conv_1')
			self.result_1_relu = tf.nn.relu(tf.nn.bias_add(self.result_1_conv, self.b_1), name='relu_1')
			self.result_1_maxpool = tf.nn.max_pool(
				value=self.result_1_relu, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool_1')
		# self.result_1 = tf.nn.lrn(input=self.result_1_maxpool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		# self.result_1 = tf.nn.dropout(x=self.result_1, keep_prob=self.dropout)

		# layer 2
		with tf.name_scope('layer_2'):
			self.w_2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 6, 16], dtype=tf.float32), name='w_2')
			self.b_2 = tf.Variable(initial_value=tf.random_normal(shape=[16], dtype=tf.float32), name='b_2')
			self.result_2_conv = tf.nn.conv2d(
				input=self.result_1_maxpool, filter=self.w_2,
				strides=[1, 1, 1, 1], padding='VALID', name='conv_2')
			self.result_2_relu = tf.nn.relu(tf.add(self.result_2_conv, self.b_2), name='relu_2')
			self.result_2_maxpool = tf.nn.max_pool(
				value=self.result_2_relu, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='VALID', name='maxpool_2')
		# self.result_2 = tf.nn.lrn(input=self.result_2_maxpool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		# self.result_2 = tf.nn.dropout(x=self.result_2, keep_prob=self.dropout)

		# expand
		with tf.name_scope('expand'):
			self.result_expand = tf.reshape(self.result_2_maxpool, [batch_size, -1])
			print(self.result_expand)

		# layer 3
		with tf.name_scope('layer_3'):
			self.w_3 = tf.Variable(initial_value=tf.random_normal(shape=[400, 120], dtype=tf.float32), name='w_4')
			self.b_3 = tf.Variable(initial_value=tf.random_normal(shape=[120], dtype=tf.float32), name='b_4')
			self.result_3_fc = tf.matmul(self.result_expand, self.w_3, name='result_4_multiply')
			self.result_3_relu = tf.nn.relu(tf.add(self.result_3_fc, self.b_3), name='relu_4')
			self.result_3_dropout = tf.nn.dropout(x=self.result_3_relu, keep_prob=self.keep_prob, name='dropout_4')

		# layer 4
		with tf.name_scope('layer_4'):
			self.w_4 = tf.Variable(initial_value=tf.random_normal(shape=[120, 84], dtype=tf.float32), name='w_4')
			self.b_4 = tf.Variable(initial_value=tf.random_normal(shape=[84], dtype=tf.float32), name='b_4')
			self.result_4_fc = tf.nn.bias_add(tf.matmul(self.result_3_dropout, self.w_4), self.b_4, name='result_4_fc')
			self.result_4_relu = tf.nn.relu(self.result_4_fc, name='result_4_relu')
			self.result_4_dropout = tf.nn.dropout(x=self.result_4_relu, keep_prob=self.keep_prob, name='dropout_4')

		# layer 5
		with tf.name_scope('layer_5'):
			self.w_5 = tf.Variable(initial_value=tf.random_normal(shape=[84, 10], dtype=tf.float32), name='w_5')
			self.b_5 = tf.Variable(initial_value=tf.random_normal(shape=[10], dtype=tf.float32), name='b_5')
			self.result_5_fc = tf.add(tf.matmul(self.result_4_dropout, self.w_5), self.b_5, name='result_5_fc')
			self.result_5_relu = tf.nn.relu(self.result_5_fc, name='result_5_relu')

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

	def train_the_model(self, train_files_queue):
		# check point
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
			# tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			# example, label = sess.run([train_images, train_labels])
			# result = sess.run(self.train_step, feed_dict={self.input_image: example, self.input_label: label})
			# print(result)
			# example, label = sess.run([train_images, train_labels])
			# print(label)
			try:
				epoch, index = 1, 1
				while not coord.should_stop():
					# Run training steps or whatever
					print('epoch ' + str(epoch))
					example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# print(label)
					# lo, acc, summary_str = sess.run(
					# 	[self.loss_mean, self.accuracy, merged_summary],
					# 	feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 1.0}
					# )
					# summary_writer.add_summary(summary_str, index)
					_, summary_str = sess.run(
						[self.train_step, merged_summary],
						feed_dict={self.input_image: example, self.input_label: label, self.keep_prob: 0.5}
					)
					summary_writer.add_summary(summary_str, index)
					index += 1
					# if index % 100 == 0:
					# 	print('num %d, loss: %.6f and accuracy: %.6f' % (index, lo, acc))
					# epoch += 1
					# for i in range(31):
					# 	example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# 	# print(label)
					# 	_, lo, acc, summary_str = sess.run(
					# 		[self.train_step, self.loss_mean, self.accuracy, merged_summary],
					# 		feed_dict={self.input_image: example, self.input_label: label}
					# 	)
					# 	summary_writer.add_summary(summary_str, index)
					# 	index += 1
					# 	if index % 100:
					# 		print('loss: ' + str(lo) + ' and accuracy: ' + str(acc))
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done train -- epoch limit reached %d' % index)
			finally:
				# When done, ask the threads to stop.
				all_parameters_saver.save(sess=sess, save_path=ckpt_path)
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print("Done training")

	def validate_the_model(self, development_files_queue):
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		development_images, development_labels = read_image_batch(development_files_queue, DEVELOPMENT_BATCH_SIZE)
		# self.result_8_softmax = tf.nn.softmax(logits=self.result_7_relu)
		# self.correct_prediction = tf.equal(tf.argmax(self.result_8_softmax, 1), tf.argmax(self.input_label, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			# print(sess.run(self.b_7))
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					print('epoch ' + str(epoch))
					# example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# print(label)
					example, label = sess.run([development_images, development_labels])  # 在会话中取出image和label
					result = sess.run(
						fetches=self.accuracy,
						feed_dict={self.input_image: example, self.input_label: label}
					)
					print(result)
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done development -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				# all_parameters_saver.save(sess=sess, save_path=ckpt_path)
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print("Done development")

	def test_the_model(self, test_files_queue):
		ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
		development_images, development_labels = read_image_batch(test_files_queue, TEST_BATCH_SIZE)
		# self.result_8_softmax = tf.nn.softmax(logits=self.result_7_relu)
		# self.correct_prediction = tf.equal(tf.argmax(self.result_8_softmax, 1), tf.argmax(self.input_label, 1))
		# self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			# print(sess.run(self.b_7))
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			try:
				epoch = 1
				while not coord.should_stop():
					# Run training steps or whatever
					print('epoch ' + str(epoch))
					# example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
					# print(label)
					example, label = sess.run([development_images, development_labels])  # 在会话中取出image和label
					result = sess.run(
						fetches=self.accuracy,
						feed_dict={self.input_image: example, self.input_label: label, self.keep_prob:1}
					)
					print(result)
					epoch += 1
			except tf.errors.OutOfRangeError:
				print('Done test -- epoch limit reached')
			finally:
				# When done, ask the threads to stop.
				# all_parameters_saver.save(sess=sess, save_path=ckpt_path)
				coord.request_stop()
			# coord.request_stop()
			coord.join(threads)
		print("Done test")


def change_prefix(files):   # 用来吧JPEG转换为JPG
	for filename in files:
		portion = os.path.splitext(filename)  # 分离文件名字和后缀
		# print(portion)
		if portion[1] == ".JPEG":  # 根据后缀来修改,如无后缀则空
			newname = portion[0] + ".jpg"  # 要改的新后缀
			os.rename(filename, newname)


def write_img_to_tfrecords():
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	# development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'development_set.tfrecords'))
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	# train set
	for index in range(10):
		images_path = glob.glob(os.path.join(FLAGS.data_dir, 'trainimage/pic2/%d/*.bmp' % index))
		print(len(images_path))
		for image_path in enumerate(images_path):
			# print(image_path[1])
			img = Image.open(image_path[1]).convert('L')
			img = img.resize((IMG_WIDE, IMG_HEIGHT))
			img_raw = img.tobytes()  # 将图片转化为二进制格式
			example = tf.train.Example(features=tf.train.Features(feature={
				"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
			}))  # example对象对label和image数据进行封装
			train_set_writer.write(example.SerializeToString())  # 序列化为字符串
	train_set_writer.close()
	print('Done train_set write')

	# test set
	for index in range(10):
		images_path = glob.glob(os.path.join(FLAGS.data_dir, 'testimage/pic2/%d/*.bmp' % index))
		print(len(images_path))
		for image_path in enumerate(images_path):
			img = Image.open(image_path[1]).convert('L')
			img = img.resize((IMG_WIDE, IMG_HEIGHT))
			img_raw = img.tobytes()  # 将图片转化为二进制格式
			example = tf.train.Example(features=tf.train.Features(feature={
				"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
			}))  # example对象对label和image数据进行封装
			test_set_writer.write(example.SerializeToString())  # 序列化为字符串
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
	min_after_dequeue = 2000
	capacity = 4000
	# image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)
	one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
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
	# train_images, train_labels = read_image_batch(train_image_filename_queue, 80)

	# development_image_filename_queue = tf.train.string_input_producer(
	# 	tf.train.match_filenames_once(development_file_path), num_epochs=1, shuffle=True)
	# development_images, development_labels = read_image_batch(development_image_filename_queue, 5)

	test_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=True)
	# test_images, test_labels = read_image_batch(test_image_filename_queue, 5)

	# net
	net = Net()
	# net.set_up_network(TRAIN_BATCH_SIZE)
	# net.train_the_model(train_image_filename_queue)
	net.set_up_network(TEST_BATCH_SIZE)
	net.test_the_model(test_image_filename_queue)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 数据地址
	parser.add_argument(
		'--data_dir', type=str, default='../input-data/MNIST/pictures',
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
	main()
