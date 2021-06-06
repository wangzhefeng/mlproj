.. _header-n2:

SARIMA
======

.. _header-n3:

SARIMA() 模型
-------------

该模型适用于含有趋势 (trend) 或季节性 (seasonal) 因素的单变量时间序列

.. code:: python

   from statsmodels.tsa.statspace.sarima import SARIMAX
   from random import random

   data = [x + random() for x in range(1, 100)]

   model = SARIMAX(data, order = (1, 1, 1), seasonal_order = (1, 1, 1, 1))
   model_fit = model.fit(disp = False)

   y_hat = model_fit.predict(len(data), len(data))
   print(y_hat)
