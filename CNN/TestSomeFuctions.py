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


class A:
	val_1, val_2 = 1, 2

	def __init__(self):
		self.val_1 = 3
		self.val_2 = 4
		print('new A')

	def print_val(self):
		print(self.val_1)

	@classmethod
	def print_val_class(cls):
		print(cls.val_1)

	@staticmethod
	def print_val_static():
		print('static')

a = A()
a.print_val()
a.print_val_class()
A.print_val_class()
A.print_val_static()
