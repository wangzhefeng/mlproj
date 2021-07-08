.. _header-n0:

Numpy-Perceptron
================

步骤：

-  定义网络结构

   -  指定输入层、隐藏层、输出层的大小

-  初始化模型参数

-  循环操作：执行前向传播 => 计算当前损失 => 执行反向传播 => 权值更新

   -  执行前向传播

   -  计算当前损失

   -  执行反向传播

   -  权值更新

.. code:: python

   """
   Using numpy build a 感知机
   """

   import numpy as np

.. _header-n25:

1.定义神经网络结构
------------------

.. _header-n26:

1 .1指定输入层、隐藏层、输出层的大小
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # data
   x_train = []
   y_train = []
   x_test = []
   y_test = []

.. _header-n28:

1.2 定义激活函数
~~~~~~~~~~~~~~~~

.. code:: python

   def sigmoid(x):
       """
       sigmoid function
       """
       y = 1 / (1 + np.exp(-x))
       return y

.. _header-n30:

2.对模型参数进行初始化
----------------------

.. code:: python

   def initilize_with_zeros(dim):
       """
       Returns:
           w: 
           b: 0
       """
       w = np.zeros((dim, 1))
       b = 0.0
       # assert (w.shape == (dim, 1))
       # assert (isinstance(b, float) or isinstance(b, int))
       return w, b

.. _header-n32:

3.循环操作：
------------

1. 执行前向传播

2. 计算当前前向传播过程的损失

3. 执行后向传播

4. 计算损失函数的梯度

5. 权值更新

.. _header-n44:

3.2 前向传播
~~~~~~~~~~~~

前向计算:

.. code:: python

   def forward(w, b, x):
       # forward
       m = x.shape[1]
       A = sigmoid(np.dot(w.T, x) + b)
       return m, A

定义损失函数:

.. code:: python

   def loss_func(m, y, y_hat):
       """
       # cross entropy loss
       """
       cost = -1 / m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
       cost = np.squeeze(cost)
       return cost

计算损失函数的梯度:

.. code:: python

   def loss_gradient_func(m, A, x, y):
       # loss function gradient
       dw = np.dot(x, (A - y).T) / m
       db = np.sum(A - y) / m
       return dw, db

前向传播过程:

.. code:: python

   def propagate(w, b, x, y):
       # forward
       m, A = forward(w, b, x)
       # cost
       cost = loss_func(m, y, A)
       # loss function gradient
       dw, db = loss_gradient_func(m, A, x, y)
       grads = {
           "dw": dw,
           "db": db
       }
       return grads, cost

.. _header-n53:

3.3 后向传播
~~~~~~~~~~~~

梯度下降算法：

.. code:: python

   def gradient_descent(w, b, dw, db, learning_rate = 0.01):
       w = w - learning_rate * dw
       b = b - learning_rate * db
       return w, b

后向传播过程：

.. code:: python

   def backward_propagation(w, b, x, y, num_iterations, learning_rate, print_cost = False):
       cost = []
       for i in range(num_iterations):
           # 前向传播
           grad, cost = propagate(w, b, x, y)
           dw = grad["dw"]
           db = grad["db"]
           # 更新权重参数
           w, b = gradient_descent(w, b, dw, db, learning_rate)
           # 保存每次迭代的损失
           if i % 100 == 0:
               cost.append(cost)
           if print_cost and i % 100 == 0:
               print("cost after iteration %i: %f" % (i, cost))
       # 更新后的权重参数
       params = {
           "w": w,
           "w": b
       }
       grads = {
           "dw": dw,
           "db": db
       }
       return params, grads, costs

.. _header-n58:

4.模型训练、评估、预测
----------------------

.. code:: python

   class model():
       
       def __init__(self):
           pass
       
       def training(self, x_train, y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
           # 权重、偏置参数初始化
           w, b = initilize_with_zeros(x_train.shape[0])
           # 模型训练：前向传播 => 计算损失 => 后向传播(计算损失函数梯度 => 更新参数)
           parameters, grads, costs = backward_propagation(w, b, x_train, y_train, num_iterations, 
                                                           learning_rate, print_cost)
           w = parameters["w"]
           b = parameters["b"]
           dw = grads["dw"]
           db = grads["db"]
           y_pred_train = evaluting(w, b, x_train, y_train)
           return w, b, dw, db, costs, y_pred_train

       def evaluting(self, w, b, x, y):
           # 前向传播
           w = w.reshape(x.shape[0], 1)
           m, A = forward(w, b, x)
           y_pred = np.zeros((1, m))
           for i in range(A.shape[1]):
               if A[:, i] > 0.5:
                   y_pred[:, i] = 1
               else:
                   y_pred[:, i] = 0
           assert(y_pred.shape == (1, m))
           print("Accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y)) * 100))
           return y_pred

       def predicting(self, w_train, b_train, x_test, y_test):
           y_pred_test = evaluting(w_train, b_train, x_test, y_test):

           return y_pred_test

       def fit(x_train, y_train, um_iterations = 2000, learning_rate = 0.5, print_cost = False):
           w, b, dw, db, costs, y_pred_train = training(x_train, y_train, num_iterations, learning_rate, print_cost)
           self.w = w
           self.b = b
           self.y_pred_train = evaluting(self.w, self.b, x_train, y_train)

       def predict(x_test, y_test):
           y_pred_test = evaluting(self.w, self.b, x_test, y_test)
           return y_pred_test
      

   model = model()
   model.fit(x_train, y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False)
   y_pred = model.predict(x_test, y_test)
