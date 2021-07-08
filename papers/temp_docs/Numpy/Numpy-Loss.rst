.. _header-n0:

Numpy-Loss
==========

.. _header-n3:

cross entropy
-------------

.. code:: python

   # 定义损失函数
   def loss_func(m, y, y_hat):
   	cost = -1 / m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
   	return cost
