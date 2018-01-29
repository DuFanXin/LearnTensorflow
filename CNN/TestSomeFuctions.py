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
import glob
import os
import random
import re
# import matplotlib.pyplot as plt
# img = Image.open('../input-data/MNIST/pictures/trainimage/pic2/0/3113.bmp')
# fig = plt.figure()
# plt.imshow(img, cmap = 'binary')#黑白显示
# plt.show()
# img.show()


a = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[5, 28, 28, 1024]))

w = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[2, 2, 512, 1024]))
b = tf.Variable(initial_value=tf.constant(value=1, dtype=tf.float32, shape=[512]))
result_up = tf.nn.conv2d_transpose(
				value=a, filter=w,
				output_shape=[5, 56, 56, 512], strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
result_conv = tf.nn.relu(tf.nn.bias_add(result_up, b, name='add_bias'), name='relu_3')
# begins = [(a.shape[0] - b.shape[0]) // 2, (a.shape[1] - b.shape[1]) // 2, 0]
# sizes = [b.shape[0], b.shape[1], b.shape[2]]
# a_ = tf.slice(
# 	input_=a,
# 	begin=[0, (tf.shape(a)[1] - tf.shape(b)[1]) // 2, (tf.shape(a)[2] - tf.shape(b)[2]) // 2, 0],
# 	size=[tf.shape(b)[0], tf.shape(b)[1], tf.shape(b)[2], tf.shape(b)[3]])
# c = tf.concat(values=[a_, b], axis=3)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print(sess.run(result_conv).shape)
