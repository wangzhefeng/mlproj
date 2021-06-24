.. _header-n0:

SoftmaxRegression_complex
=========================

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-


   import d2lzh as d2l
   from mxnet.gluon import data as gdata
   from mxnet import autograd, nd
   import sys
   import time

   # #############################################################
   # data
   # #############################################################
   mnist_train = gdata.vision.FashionMNIST(train = True)
   mnist_test = gdata.vision.FashionMNIST(train = False)

   def get_fashion_mnist_labels(labels):
   	text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", 
   				   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
   	return [text_labels[int(i)] for i in labels]

   def show_fashion_mnist(images, labels):
   	d2l.use_svg_display()
   	_, figs = d2l.plt.subplots(1, len(images), figsize = (12, 12))

   # #############################################################
   # 读取数据集
   # #############################################################
   batch_size = 256
   train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


   # #############################################################
   # 初始化模型参数
   # #############################################################
   num_inputs = 784
   num_outputs = 10
   W = nd.random.normal(scale = 0.01, shape = (num_inputs, num_outputs))
   b = nd.zeros(num_outputs)
