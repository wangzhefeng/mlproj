.. _header-n0:

ARIMA
=====

.. _header-n3:

ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型
---------------------------------------------

Autoregressive Integrated Moving Average (ARIMA),
差分自回归移动平均模型，是差分后的时间序列和残差误差的线性函数.

差分运算具有强大的确定性信息提取能力，许多非平稳序列差分后会显示出平稳序列的性质，称这个非平稳序列为差分平稳序列，对差分平稳序列可以使用
ARIMA(autoregression integrated moving average, 求和自回归移动平均)
模型进行拟合.

ARIMA 模型的实质就是差分运算和 ARMA
模型的组合，说明任何非平稳序列如果能通过适当阶数的差分实现差分后平稳，就可以对差分后序列进行
ARMA 模型拟合，而 ARMA 模型的分析方法非常成熟.

.. _header-n7:

ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型结构
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \left\{
   \begin{array}{**lr**}
   \Phi(B)\Delta^{d}x_{t} = \Theta(B)\epsilon_{t}& \\
   E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
   E(x_{s}\epsilon_{t}) = 0, \forall s < t&
   \end{array}
   \right.

其中：

-  :math:`{\epsilon_{t}}` 为零均值白噪声序列

-  :math:`\Delta^{d} = (1-B)^{d}`

-  :math:`\Phi(B) = 1-\sum_{i=1}^{p}\phi_{i}B^{i}` 为平稳可逆
   ARMA(\ :math:`p`, :math:`q`) 模型的自回归系数多项式

-  :math:`\Theta(B) = 1 + \sum_{i=1}^{q}\theta_{i}B^{i}` 为平稳可逆
   ARMA(\ :math:`p`, :math:`q`) 模型的移动平滑系数多项式

ARIMA 之所以叫 **求和自回归移动平均** 是因为：\ :math:`d`
阶差分后的序列可以表示为下面的表示形式，即差分后序列等于原序列的若干序列值的加权和，而对它又可以拟合
ARMA 模型：

:math:`\Delta^{d}x_{t} = \sum_{i=0}^{d}(-1)C_{d}^{i}x_{t-i}, 其中：C_{d}^{i} = \frac{d!}{i!(d-i)!}`

ARIMA 模型的另一种形式：

:math:`\Delta^{d}x_{t} = \frac{\Theta(B)}{\Phi(B)}\epsilon_{t}`

其中：

-  当 :math:`d=0` 时 ARIMA(\ :math:`p`, :math:`0`, :math:`q`) 模型就是
   ARMA(\ :math:`p`, :math:`q`) 模型

-  当 :math:`p=0` 时，ARIMA(\ :math:`0`, :math:`d`, :math:`q`)
   模型可以简记为 IMA(\ :math:`d`, :math:`q`) 模型

-  当 :math:`q=0` 时，ARIMA(\ :math:`p`, :math:`d`, :math:`0`)
   模型可以简记为 ARI(\ :math:`p`, :math:`d`) 模型

-  当 :math:`d=1, p=q=0` 时，ARIMA(\ :math:`0`, :math:`1`, :math:`0`)
   模型为 随机游走 (random walk) 模型:

.. math::

   \left\{
   \begin{array}{**lr**}
   x_{t} = x_{t-1} + \epsilon_{t}& \\
   E(\epsilon_{t}) =0, Var(\epsilon_{t}) = \sigma_{\epsilon}^{2}, E(\epsilon_{s}\epsilon_{t}) = 0, s \neq t& \\
   E(x_{s}\epsilon_{t}) = 0, \forall s < t&
   \end{array}
   \right.

.. _header-n34:

ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型的统计性质
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 平稳性

2. 方差齐性

.. _header-n40:

ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型建模
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 获得时间序列观察值

2. 平稳性检验

   -  不平稳：差分运算 => 平稳性检验

   -  平稳：下一步

3. 白噪声检验

   -  不通过：拟合 ARMA 模型 => 白噪声检验

   -  通过：分析结束

.. _header-n58:

ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型应用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from statsmodels.tsa.arima_model import ARIMA
   from random import random

   data = [x + random() for x in range(1, 100)]

   model = ARIMA(data, order = (1, 1, 1))
   model_fit = model.fit(disp = True)

   y_hat = model_fit.predict(len(data), len(data), typ = "levels")
   print(y_hat)
