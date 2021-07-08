.. _header-n0:

MLP_simple
==========

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-
   # @Date    : 2019-07-30 20:57:52
   # @Author  : Your Name (you@example.org)
   # @Link    : http://example.org
   # @Version : $Id$

   import d2lzh as d2l
   from mxnet.gluon import nn
   from mxnet import gluon, init
   from mxnet.gluon import loss as gloss


   # ###################################################
   # 读取数据集
   # ###################################################
   batch_size = 256
   train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


   # ###################################################
   # 定义模型
   # ###################################################
   net = nn.Sequential()
   net.add(nn.Dense(256, activation = "relu"), 
   		nn.Dense(10))
   net.initialize(init.Normal(sigma = 0.01))


   # ###################################################
   # 训练模型
   # ###################################################
   loss = gloss.SoftmaxCrossEntropyLoss()
   trainer = gluon.Trainer(net.collect_param(), 
   						"sgd", 
   						{"learning_rate": 0.5})
   num_epochs = 5
   d2l.train_ch3(net, 
   			  train_iter,
   			  test_iter,
   			  loss, 
   			  num_epochs, 
   			  batch_size,
   			  None,
   			  None,
   			  trainer)
