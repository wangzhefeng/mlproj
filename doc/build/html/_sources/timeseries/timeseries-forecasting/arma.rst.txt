.. _header-n2:

ARMA
====

.. _header-n3:

ARMA(\ :math:`p`, :math:`q`) 模型
---------------------------------

ARMA(\ :math:`p`,
:math:`q`)，自回归移动平均模型，是时间序列和残差误差的线性函数，是
AR(\ :math:`p`) 和 MA(\ :math:`q`) 模型的组合，该模型适用于无趋势
(trend) 和季节性 (seasonal) 因素的单变量时间序列

.. _header-n5:

ARMA(\ :math:`p`, :math:`q`) 模型结构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \left\{
   \begin{array}{**lr**}
   x_{t}=\phi_{0} + \phi_{1}x_{t-1} + \phi_{2}x_{t-2} + \cdots + \phi_{p}x_{t-p} + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \theta_{2}\epsilon_{t-2} + \cdots + \theta_{q}\epsilon_{t-q} & \\
   \phi_{p} \neq 0, \theta_{q} \neq 0& \\
   E(\epsilon_{t}) = 0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{t}\epsilon_{s}) = 0, s \neq t & \\
   E(x_{s}\epsilon_{t}) = 0, \forall s < t & 
   \end{array}
   \right.

ARMA 模型的另一种形式：

:math:`(1-\sum_{i=1}^{p}\phi_{i}B^{i})x_{t} = (1 - \sum_{i=1}^{q}\theta_{i}B^{i})\epsilon_{t}`

:math:`\Phi(B)x_{t} = \Theta(B)\epsilon_{t}`

-  当 :math:`q = 0` 时, ARMA(\ :math:`p`, :math:`q`) 模型就退化成了
   AR(\ :math:`p`) 模型.

-  当 :math:`p = 0` 时, ARMA(\ :math:`p`, :math:`q`) 模型就退化成了
   MA(\ :math:`q`) 模型.

-  所以 AR(\ :math:`p`) 和 MA(\ :math:`q`) 实际上是 ARMA(\ :math:`p`,
   :math:`p`) 模型的特例, 它们统称为 ARMA 模型. 而 ARMA(\ :math:`p`,
   :math:`p`) 模型的统计性质也正是 AR(\ :math:`p`) 模型和
   MA(\ :math:`p`) 模型统计性质的有机结合.

.. _header-n17:

ARMA(\ :math:`p`, :math:`q`) 模型的统计性质
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  均值

-  自协方差函数

-  自相关系数

.. _header-n25:

ARMA(\ :math:`p`, :math:`q`) 模型的应用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from statsmodels.tsa.arima_model import ARMA
   from random import random

   data = [random.() for x in range(1, 100)]

   model = ARMA(data, order = (2, 1))
   model_fit = model.fit(disp = False)

   y_hat = model_fit.predict(len(data), len(data))
   print(y_hat)
