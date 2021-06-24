.. _header-n0:

LinearRegression_simple
=======================

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-


   from mxnet import nd, autograd
   from mxnet.gluon import data as gdata
   from mxnet.gluon import nn
   from mxnet import init
   from mxnet.gluon import loss as gloss
   from mxnet import gluon


   # #######################################################################
   # 生成数据集
   # #######################################################################
   num_inputs = 2
   num_examples = 1000
   true_w = [2, -3.4]
   true_b = 4.2
   features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))
   labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
   labels += nd.random.normal(scale = 1, shape = labels.shape)


   # #######################################################################
   # 读取数据集
   # #######################################################################
   batch_size = 10
   dataset = gdata.ArrayDataset(features, labels)
   data_iter = gdata.DataLoader(dataset, batch_size, shuffle = True)


   # #######################################################################
   # 定义模型
   # #######################################################################
   net = nn.Sequential()
   net.add(nn.Dense(1))


   # #######################################################################
   # 初始化模型参数
   # #######################################################################
   net.initialize(init.Normal(sigma = 0.01))



   # #######################################################################
   # 定义损失函数
   # #######################################################################
   loss = gloss.L2Loss()


   # #######################################################################
   # 定义优化算法
   # #######################################################################
   trainer = gluon.Trainer(net.collect_params(), 
   						"sgd", 
   						{"learning_rate": 0.03})


   # #######################################################################
   # 训练模型
   # #######################################################################
   num_epochs = 3
   for epoch in range(1, num_epochs + 1):
   	for X, y in data_iter:
   		with autograd.record():
   			l = loss(net(X), y)
   		l.backward()
   		trainer.step(batch_size)
   	l = loss(net(features), labels)
   	print("epoch %d, loss %f" % (epoch, l.mean().asnumpy()))

   dense = net[0]

   print("生成数据使用的权重参数：", true_w)
   print("模型训练生成的权重参数：", dense.weight.data())
   print("生成数据使用的偏置参数：", true_b)
   print("模型训练生成的偏置参数：", dense.bias.data())
