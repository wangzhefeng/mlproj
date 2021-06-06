.. _header-n0:

传统时间序列建模方法
====================

.. _header-n3:

1.时间序列介绍
--------------

.. _header-n4:

1.1 时间序列的定义
~~~~~~~~~~~~~~~~~~

   -  时间序列是指在一定时间内按时间顺序测量的某个变量的取值序列.

   -  时间序列分析就是使用统计的手段对这个序列的过去进行分析，以此对该变量的变化特性建模、并对未来进行预测.

   -  时间序列分析试图通过研究过去来预测未来.

.. _header-n15:

1.2 时间序列的性质
~~~~~~~~~~~~~~~~~~

时间序列的性质:

   - 自相关

   - 平稳性

      -  如何检验稳定性

      -  如何使得序列稳定

   - 季节性

   - 趋势性

   - 随机噪声


-  自相关

   -  自相关是时间序列观测值之间的相似度，是观测值之间时间滞后的函数

-  季节性

   -  季节性是指周期性的波动

-  平稳性

   - 平稳性是指如果时间序列的统计性质不随时间变化，则称该时间序列是平稳的，即：该时间序列有不变的均值和方差，协方差不随时间变化。通常，股票价格不是一个平稳的过程，因为可能看到一个增长的趋势，或者，其波动性会随着时间的推移而增加(方差在变)

   - 理想情况下，需要一个用于建模的固定时间序列。当然，不是所有的时间序列都是平稳的，但是可以通过做不同的变换使它们保持平稳

   - 如何检测过程是否平稳?

      -  Dickey-Fuller 检验可以检验时间序列是否是平稳的

         -  p>0，过程不是平稳的

         -  p=0，过程是平稳的

.. _header-n45:

1.3 时间序列分析方法
~~~~~~~~~~~~~~~~~~~~

-  描述性时间序列分析

-  统计时间序列分析

.. _header-n51:

2.时间序列的预处理
------------------

.. _header-n52:

2.1 平稳性检验
~~~~~~~~~~~~~~

.. _header-n54:

2.2 纯随机性检验
~~~~~~~~~~~~~~~~

.. _header-n56:

3.时间序列建模
--------------

   时间序列预测的问题, 并不是普通的回归问题, 而是 **自回归**.

-  一般的回归问题比如最简单的线性回归模型:
   :math:`Y=a \cdot X_1+b \cdot X_2`, 讨论的是因变量 :math:`Y`
   关于两个自变量 :math:`X_1`\ 和 :math:`X_2` 的关系, 目的是找出最优的
   :math:`a` 和 :math:`b` 来使预测值 :math:`y=a \cdot X_1+b \cdot X_2`
   逼近真实值 :math:`Y`.

-  自回归模型中, 自变量 :math:`X_1` 和 :math:`X_2` 都为 :math:`Y` 本身,
   也就是说 :math:`Y(t)=a \cdot Y(t-1)+ b \cdot Y(t-2)`,其中
   :math:`Y(t-1)` 为 :math:`Y` 在 :math:`t-1` 时刻的值, 而
   :math:`Y(t-2)` 为 :math:`Y` 在 :math:`t-2` 时刻的值, 换句话说, 现在的
   :math:`Y` 值由过去的 :math:`Y` 值决定, 因此自变量和因变量都为自身,
   因此这种回归叫自回归.

-  自回归模型都有着严格理论基础,讲究时间的平稳性,
   需要对时间序列进行分析才能判断是否能使用此类模型.
   这些模型对质量良好的时间序列有比较高的精度。传统的自回归模型有：

   -  移动平均

   -  指数平滑

   -  自回归模型（AR）

   -  移动平均模型（MA）

   -  自回归移动平均模型（ARMA）

   -  差分自回归移动平均模型（ARIMA）

.. _header-n80:

3.1 平稳时间序列分析
~~~~~~~~~~~~~~~~~~~~

一个时间序列经过预处理被识别为
**平稳非白噪声时间序列**\ ，就说明该序列是一个蕴含相关信息的平稳序列。在统计上，通常是建立一个线性模型来拟合该序列的发展，借此提取该序列中的有用信息。ARMA(auto
regression moving average)模型就是目前最常用的平稳序列拟合模型.

假设一个时间序列经过预处理被识别为 **平稳非白噪声时间序列**,就可以利用
ARMA 模型对该序列建模. 建模的基本步骤为:

1. 求出该观察序列的 **样本自相关系数(ACF)** 和
   **样本偏自相关系数(PACF)** 的值;

2. 根据样本自相关系数和偏自相关系数的性质,选择阶数适当的
   ARMA(\ :math:`p`,\ :math:`q`)模型进行拟合;

3. 估计模型中的位置参数的值;

4. 检验模型的有效性;

   -  如果拟合模型通不过检验,转向步骤2,重新选择模型再拟合

5. 模型优化.

   -  如果拟合模型通过检验,仍然转向步骤2,充分考虑各种可能,建立多个拟合模型,从所有通过检验的拟合模型中选择最优模型

6. 利用拟合模型,预测序列的将来走势.

.. _header-n103:

3.1.1 差分
^^^^^^^^^^

**差分运算:**

:math:`p` 阶差分：相距一期的两个序列值至之间的减法运算称为 :math:`1`
阶差分运算；对 :math:`1` 阶差分后序列在进行一次 :math:`1` 阶差分运算称为
:math:`2` 阶差分；以此类推，对 :math:`p-1` 阶差分后序列在进行一次
:math:`1` 阶差分运算称为 :math:`p` 阶差分.

:math:`\Delta x_{t} = x_{t-1} - x_{t-1}`

:math:`\Delta^{2} x_{t} = \Delta x_{t} - \Delta x_{t-1}`

:math:`\Delta^{p} x_{t} = \Delta^{p-1} x_{t} - \Delta^{p-1} x_{t-1}`

:math:`k` 步差分：相距 :math:`k` 期的两个序列值之间的减法运算称为
:math:`k` 步差分运算.

:math:`\Delta_{k}x_{t} = x_{t} - x_{t-k}`

**滞后算子：**

滞后算子类似于一个时间指针，当前序列值乘以一个滞后算子，就相当于把当前序列值的时间向过去拨了一个时刻.

假设 :math:`B` 为滞后算子:

:math:`x_{t-1} = Bx_{t}`

:math:`x_{t-2} = B^{2}x_{t}`

:math:`\vdots`

:math:`x_{t-p} = B^{p}x_{t}`

也可以用滞后算子表示差分运算:

-  :math:`p` 阶差分

   -  :math:`\Delta^{p}x_{t} = (1-B)^{p}x_{t} = \sum_{i=0}^{p}(-1)C_{p}^{i}x_{t-i}`

-  :math:`k` 步差分

   -  :math:`\Delta_{k}x_{t} = x_{t} - x_{t-k} = (1-B^{k})x_{t}`

**线性差分方程：**

.. _header-n132:

3.1.2 ARMA模型
^^^^^^^^^^^^^^

ARMA 模型的全称是
**自回归移动平均模型**,它是目前最常用的拟合平稳序列的模型.它又可以细分为
AR 模型, MA 模型, ARMA 模型三类.

**AR(\ :math:`p`) 模型**

AR(\ :math:`p`) 模型结构:

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

AR(\ :math:`p`) 模型的统计性质:

-  均值

-  方差

-  自协方差函数

-  自相关系数

-  偏自相关系数

**MA(\ :math:`q`) 模型**

MA(\ :math:`q`) 模型结构:

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

MA(\ :math:`q`) 模型的统计性质:

-  常数均值

-  常数方差

-  自协方差函数只与滞后阶数相关，且 :math:`q` 阶截尾

-  自相关系数 :math:`q` 阶截尾

**ARMA(\ :math:`p`, :math:`q`) 模型**

ARMA(\ :math:`p`, :math:`q`) 模型结构:

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

ARMA(\ :math:`p`, :math:`q`) 模型的统计性质:

-  均值

-  自协方差函数

-  自相关系数

.. _header-n196:

3.2 非平稳时间序列分析
~~~~~~~~~~~~~~~~~~~~~~

在自然界中绝大部分序列都是非平稳的,因而对非平稳序列的分析更普遍、更重要.
对非平稳时间序列分析方法可以分为 **随机时间序列分析** 和
**确定性时间序列分析**.

.. _header-n199:

3.2.1 时间序列的分解
^^^^^^^^^^^^^^^^^^^^

.. _header-n201:

3.2.2 差分运算
^^^^^^^^^^^^^^

.. _header-n203:

3.2.3 ARIMA 模型
^^^^^^^^^^^^^^^^

差分运算具有强大的确定性信息提取能力，许多非平稳序列差分后会显示出平稳序列的性质，称这个非平稳序列为差分平稳序列，对差分平稳序列可以使用
ARIMA(autoregression integrated moving average, 求和自回归移动平均)
模型进行拟合.

ARIMA 模型的实质就是差分运算和 ARMA
模型的组合，说明任何非平稳序列如果能通过适当阶数的差分实现差分后平稳，就可以对差分后序列进行
ARMA 模型拟合，而 ARMA 模型的分析方法非常成熟.

**ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型结构:**

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

**ARIMA(\ :math:`p`, :math:`d`, :math:`q`) 模型的统计性质:**

1. 平稳性

2. 方差齐性

**ARIMA 模型建模:**

1. 获得时间序列观察值

2. 平稳性检验

   -  不平稳：差分运算 => 平稳性检验

   -  平稳：下一步

3. 白噪声检验

   -  不通过：拟合 ARMA 模型 => 白噪声检验

   -  通过：分析结束

.. _header-n259:

3.3 多元时间序列分析
~~~~~~~~~~~~~~~~~~~~

.. _header-n266:

3.3.1 移动平均
^^^^^^^^^^^^^^

-  移动平均模型是最简单的时间序列建模方法，即：下一个值是所有一个时间窗口中值的平均值

-  时间窗口越长，预测值的趋势就越平滑

.. _header-n273:

3.3.2 指数平滑
^^^^^^^^^^^^^^

-  指数平滑使用了与移动平均相似的逻辑，但是，指数平滑对每个观测值分配了不同的递减权重，即：离现在的时间距离越远，时间序列观测值的重要性就越低

-  指数平滑的数学表示：

:math:`y=\alpha x_{t} + (1 - \alpha)y_{t-1}, t>0`

其中:

-  :math:`\alpha \in [0, 1]`
   是一个平滑因子，决定了之前观测值的权重下降的速度。平滑因子越小，时间序列就越平滑，因为当平滑因子接近0时，指数平滑接近移动平均模型

.. _header-n284:

3.3.2.1 双指数平滑
''''''''''''''''''

-  当时间序列中存在趋势时，使用双指数平滑，它只是指数平滑的两次递归使用

-  双指数平滑的数学表示：

:math:`y=\alpha x_{t} + (1 - \alpha)(y_{t-1} + b_{t-1})`
:math:`b_{t}=\beta (y_{t} - y_{t-1}) + (1 - \beta)b_{t-1}`

其中:

-  :math:`\alpha \in [0, 1]` 是一个平滑因子

-  :math:`\beta \in [0, 1]` 是趋势平滑因子

.. _header-n298:

3.3.2.2 三指数平滑
''''''''''''''''''

-  三指数平滑通过添加季节平滑因子扩展双指数平滑

-  三指数平滑的数学表示：

:math:`y=\alpha \frac{x_{t}}{c_{t-L}} + (1 - \alpha)(y_{t-1} + b_{t-1})`
:math:`b_{t}=\beta (y_{t} - y_{t-1}) + (1 - \beta)b_{t-1}`
:math:`c_{t}=\gamma \frac{x_{t}}{y_{t}} + (1-\gamma)c_{t-L}`

其中:

-  :math:`\alpha \in [0, 1]` 是一个平滑因子

-  :math:`\beta \in [0, 1]` 是趋势平滑因子

-  :math:`\gamma` 是季节长度
