.. _header-n2:

AR
==

.. _header-n3:

AR(\ :math:`p`) 模型
--------------------

AR(\ :math:`p`)，\ :math:`p` 阶自回归模型，该模型适用于无趋势 (trend)
和季节性 (seasonal) 因素的单变量时间序列

.. _header-n5:

AR(\ :math:`p`) 模型结构:
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \left\{
   \begin{array}{**lr**}
   x_{t}=\phi_{0} + \phi_{1}x_{t-1} + \phi_{2}x_{t-2} + \cdots + \phi_{p}x_{t-p} + \epsilon_{t} & \\
   \phi_{p} \neq 0 & \\
   E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
   E(x_{s}\epsilon_{t}) = 0, \forall s < t & 
   \end{array}
   \right.

其中：

-  :math:`\epsilon_{t}` 是白噪声序列

-  :math:`\phi_{0}` 是常数，表示时间序列没有进行 0 均值化

.. _header-n14:

AR(\ :math:`p`) 模型的统计性质
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  均值

-  方差

-  自协方差函数

-  自相关系数

-  偏自相关系数

.. _header-n26:

AR(\ :math:`p`) 模型应用
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from statsmodels.tsa.ar_model import AR
   from random import random

   data = [x + random() for x in range(1, 100)]

   model = AR(data)
   model_fit = model.fit()

   y_hat = model_fit.predict(len(data), len(data))
   print(y_hat)
