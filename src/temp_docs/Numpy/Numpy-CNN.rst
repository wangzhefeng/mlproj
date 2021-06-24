.. _header-n0:

Numpy-CNN
=========

.. code:: python

   def conv_single_step(a_slice_prev, W, b):
   	s = a_slice_prev * W
   	Z = np.sum(s)
   	Z = float(Z + b)
   	return Z
