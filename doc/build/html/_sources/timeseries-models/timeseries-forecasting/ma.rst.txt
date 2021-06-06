.. _header-n2:

MA
==

.. _header-n3:

MA(\ :math:`q`) 模型
--------------------

MA(\ :math:`p`)，\ :math:`q` 阶移动平均模型，残差误差(residual erros)
的线性函数，与计算时间序列的移动平均不同，该模型适用于无趋势 (trend)
和季节性 (seasonal) 因素的单变量时间序列

.. _header-n5:

MA(\ :math:`q`) 模型结构:
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \left\{
   \begin{array}{**lr**}
   x_{t}=\mu + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q}& \\
   \theta_{q} \neq 0 & \\
   E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t &
   \end{array}
   \right.

其中：

-  :math:`\epsilon_{t}` 是白噪声序列

-  :math:`\mu` 是常数

.. _header-n13:

MA(\ :math:`q`) 模型的统计性质:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  常数均值

-  常数方差

-  自协方差函数只与滞后阶数相关，且 :math:`q` 阶截尾

-  自相关系数 :math:`q` 阶截尾

.. _header-n23:

MA(\ :math:`q`) 模型应用
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from statsmodesl.tsa.arima_model import ARMA
   from random import random

   data = [x + random() for x in range(1, 100)]

   model = ARMA(data, order = (0, 1))
   model_fit = model.fit(disp = False)

   y_hat = model_fit.predict(len(data), len(data))
   print(y_hat)
