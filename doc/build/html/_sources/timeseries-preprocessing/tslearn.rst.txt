.. _header-n0:

tslearn
=======

.. _header-n3:

tslearn 文档
------------

   tslearn is a Python package that provides machine learning tools for
   the analysis of time series. This package builds on (and hence
   depends on) scikit-learn, numpy and scipy libraries.

..

   If you plan to use the shapelets module from tslearn, keras and
   tensorflow should also be installed.

.. _header-n8:

1.依赖 Python 库
~~~~~~~~~~~~~~~~

-  ``scikit-learn``

-  ``numpy``

-  ``scipy``

-  ``keras``

-  ``tensorflow``

.. _header-n20:

2.tslean 安装
~~~~~~~~~~~~~

conda:

.. code:: shell

   $ conda install -c conda-forge tslearn

PyPI:

.. code:: shell

   $ pip install tslearn

github-hosted version:

.. code:: shell

   $ pip install git+https://github.com/rtavenar/tslearn.git

.. _header-n27:

3.用法
~~~~~~

.. _header-n28:

3.1 数据、数据处理
^^^^^^^^^^^^^^^^^^

.. code:: python

   from tslearn.utils import to_time_series
   from tslearn.utils import to_time_series_datasets

   my_first_time_series = [1, 3, 4, 2]
   my_second_time_series = [1, 2, 4, 5]
   my_third_time_series = [1, 2, 4, 2, 2]
   formatted_time_series = to_time_series(my_first_time_series)
   print(formatted_time_series)
   print(formatted_time_series.shape)

   formatted_dataset = to_time_series_dataset([my_first_time_series, my_second_time_series])
   print(formatted_dataset)
   print(formatted_dataset.shape)

   formatted_dataset = to_time_series_dataset([my_first_time_series, my_second_time_series, my_third_time_series])
   print(formatted_dataset)
   print(formatted_dataset.shape)

.. code:: 

   [[1.]
    [3.]
    [4.]
    [2.]]
   (4, 1)


   [[[1.]
     [3.]
     [4.]
     [2.]]

    [[1.]
     [2.]
     [4.]
     [5.]]]
   (2, 4, 1)


   [[[ 1.]
     [ 3.]
     [ 4.]
     [ 2.]
     [nan]]

    [[ 1.]
     [ 2.]
     [ 4.]
     [ 5.]
     [nan]]

    [[ 1.]
     [ 2.]
     [ 4.]
     [ 2.]
     [ 2.]]]
   (3, 5, 1)

.. _header-n31:

3.2 tslearn 中的数据
^^^^^^^^^^^^^^^^^^^^

tslearn 中自带的数据：

.. code:: python

   from tslearn.datasets import UCR_UEA_datasets

   X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
   print(X_train.shape)
   print(y_train.shape)

数据读取、保存：

.. code:: python

   from tslearn.utils import save_timeseries_txt, load_timeseries_txt
   time_series_dataset = load_timeseries_txt("path/to/your/file.txt")
   save_timeseries_txt("path/to/your/file.txt", dataset_to_be_saved)

对数据建模：

.. code:: python

   from tslearn.clustering import TimeSeriesKMeans

   km = TimeSeriesKMeans(n_clusters = 3, metric = "dtw")
   km.fit(X_train)

.. _header-n39:

4.Examples
~~~~~~~~~~

.. _header-n40:

4.1 DTW computation
~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np 
   import matplotlib.pyplot as plt
   from tslearn.generators import random_walks
   from tslearn.preprocessing import TimeSeriesScalerMeanVariance
   from tslearn import metrics

   np.random.seed(0)
   n_ts, sz, d = 2, 100, 1
   dataset = random_walks(n_ts = n_ts, sz = sz, d = d)
   scaler = TimeSeriesScalerMeanVariance(mu = 0, std = 1.)
   dataset_scaled = scaler.fit_transform(dataset)
   path, sim = metric 

.. _header-n42:

5.APIs
~~~~~~
