.. _header-n0:

data-preprocessing
==================

.. _header-n3:

1.创建NDArray
-------------

.. code:: python

   from mxnet import nd
   # 创建 NDArray 实例
   x = nd.arange(12)
   x
   # NDArray 实例的形状
   x.shape
   # NDArray 实例中元素的总数
   x.size
   # 把行向量转换为矩阵
   X = x.reshape((3, 4))
   X

.. _header-n5:

2.自动求梯度
------------

求 :math:`y=2x^{T}x` 关于向量 :math:`x` 的梯度：

.. code:: python

   from mxnet import autograd, nd
   import numpy as np

   x = nd.arange(4).reshape((4, 1))
   x.attach_grad()
   with autograd.record():
       y = 2 * nd.dot(x.T, x)
   y.backward()

   assert (x.grad - 4 * x).norm().asscalar() == 0
   x.grad

   print(autograd.is_training())
   with autograd.record():
       print(autograd.is_training())



   def f(a):
       b = a * 2
       while b.norm().asscalar() < 1000:
           b = b * 2
       if b.sum().asscalar() > 0:
           c = b
       else:
           c = 100 * b
       return c

   a = nd.random.normal(shape = 1)
   a.attach_grad()
   with autograd.record():
       c = f(a)
   c.backward()
   a.grad == c / a
