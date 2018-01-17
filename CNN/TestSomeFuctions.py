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
# import matplotlib.pyplot as plt
img = Image.open('../input-data/MNIST/pictures/trainimage/pic2/0/3113.bmp')
# fig = plt.figure()
# plt.imshow(img, cmap = 'binary')#黑白显示
# plt.show()
img.show()

# for i in range(10):
# 	images_path = glob.glob(os.path.join(FLAGS.data_dir, str(i) + '/*.bmp'))
# 	# print(len(images_path))
# 	file_queue = tf.train.string_input_producer(string_tensor=images_path, num_epochs=1)
# 	image_reader = tf.WholeFileReader()
# 	_, image_file = image_reader.read(file_queue)
# 	image = tf.image.decode_bmp(contents=image_file, channels=0)
# 	# key = tf.decode_raw(bytes=key, out_type=tf.uint8)
# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		sess.run(tf.local_variables_initializer())
# 		coord = tf.train.Coordinator()
# 		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
# 		print(sess.run(image))
# 		coord.request_stop()
# 		coord.join(threads)
