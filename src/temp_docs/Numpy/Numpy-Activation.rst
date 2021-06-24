.. _header-n0:

激活函数
========

.. _header-n3:

Sigmoid
-------

.. code:: python

   def sigmoid(x):
   	"""
   	sigmoid function
   	"""
   	y = 1 / (1 + np.exp(-x))
   	return y

.. _header-n5:

ReLU
----

.. code:: python

   def relu(z):
       y = np.maximum(0, x)
       return y

.. _header-n7:

tanh
----

.. code:: python

   def tanh(x):
   	pass
