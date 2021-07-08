.. _header-n0:

weight\ *decay*\ complex
========================

.. code:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-
   # @Date    : 2019-08-04 13:18:37
   # @Author  : Your Name (you@example.org)
   # @Link    : http://example.org
   # @Version : $Id$

   import d2lzh as d2l
   from mxnet import autograd, gluon, init, nd
   from mxnet.gluon import data as gdata, loss as gloss, nn


   # #################################################################
   # 构造模拟数据集
   # #################################################################
   n_train, n_test, num_inputs = 20, 100, 200
   true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

   features = nd.random.normal(shape = (n_train + n_test, num_inputs))
   labels = nd.dot(features, true_w) + true_b
   labels += nd.random.normal(scale = 0.01, shape = labels.shape)

   train_features, test_features = features[:n_train, :], features[n_train:, :]
   train_labels, test_labels = labels[:n_train], lables[n_train:]

   # #################################################################
   # 初始化模型参数
   # #################################################################
   def init_params():
       w = nd.random.normal(scale = 1, shape = (num_inputs, 1))\
       b = nd.zeros(shape = (1, ))
       w.attach_grad()
       b.attach_grad()
       return [w, b]


   # #################################################################
   # 定义L2范数惩罚项
   # #################################################################
   def l2_penalty(w):
       penalty = w ** 2.sum() / 2
       return penalty

   # #################################################################
   # 定义训练和测试
   # #################################################################
   batch_size = 1
   num_epochs = 100
   lr = 0.003
   net = d2l.linreg
   loss = d2l.squared_loss
   train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                                 batch_size,
                                 shuffle = True)

   def fit_and_plot(lambd):
       w, b = init_params()
       train_ls, test_ls = [], []
       for _ in range(num_epochs):
           for X, y in train_iter:
               with autograd.record():
                   l = loss(net(X, w, b), y) + labmd * l2_penalty(w)
               l.backward()
               d2l.sgd([w, b], lr, batch_size)
           train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
           test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
       d2l.semilogy(range(1, num_epochs + 1), train_ls, "epochs", "loss",
                    range(1, num_epochs + 1), test_ls, ["train", "test"])
       print("L2 norm of w:", w.norm().asscalar())

   # #################################################################
   # 模型表现
   # #################################################################
   # 观察过拟合
   fit_and_plot(lambd = 0)

   # 使用权重衰减
   fit_and_plot(lambd = 3)
