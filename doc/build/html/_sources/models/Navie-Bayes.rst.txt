.. _header-n0:

朴素贝叶斯(Navie Bayes)
=======================

   -  朴素贝叶斯是基于\ **贝叶斯定理**\ 和\ **特征条件独立**\ 假设的分类方法；

   -  对于给定的训练数据集：

   -  首先，基于特征条件独立假设学习输入、输出的联合概率分布；

   -  其次，基于此模型，对给定的输入\ :math:`x`\ ，利用贝叶斯定理求出后验概率最大的输出\ :math:`y`\ ；

.. _header-n16:

模型推导
--------

给定数据集：
:math:`\{(x_i, y_i)\}`\ ，其中：\ :math:`i = 1, 2, \ldots, N`\ ，\ :math:`x_i \in R^p`\ ，\ :math:`y_i \in \{c_1, c_2, \ldots, c_K\}`\ ；

假设

-  训练数据集 :math:`\{(x_i, y_i)\}, i = 1, 2, \ldots, N` 由
   :math:`P(x, y)`\ 独立同分布产生；

-  :math:`P(x, y)`\ ： 是 :math:`x` 和 :math:`y` 的联合概率分布；

-  :math:`P(y = c_k), i = 1, 2, \ldots, K`\ ：是目标变量 :math:`y`
   的先验分布；

-  :math:`P(x|y=c_k)`\ ：是给定目标变量 :math:`y=c_k` 下，预测变量
   :math:`x` 条件分布；

根据条件概率的条件独立性假设：

.. math::

   \begin{eqnarray}
   P(x|y=c_k) & & {} = P(x_{ij}|y_i=c_k) \nonumber \\
   		   & & {} = \prod_{j=1}^{p}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}

根据Bayesian定理，求解给预测变量 :math:`x` 下，目标变量
:math:`y=c_k`\ 的后验概率：

.. math::

   \begin{eqnarray}
   P(y=c_k|x) & & {} = \frac{P(x, y = c_k)}{P(x)} \nonumber \\
   		   & & {} = \frac{P(x|y=c_k)P(y=c_k)}{\sum_{k}P(x|y=c_k)P(Y=c_k)} \nonumber \\
   		   & & {} = \frac{P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k)}{\sum_k P(y_i=c_k)\prod_j P(x_{ij}|y_i=c_k)} \nonumber
   \end{eqnarray}

朴素贝叶斯分类器：

.. math::

   \begin{eqnarray}
   y_i=f(x_i) & & {} = \arg\underset{c_k}{\max} P(y=c_k|x_i) \nonumber \\
   	       & & {} = \arg\underset{c_k}{\max} P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}

.. _header-n36:

模型学习方法
------------

.. _header-n37:

极大似然估计
~~~~~~~~~~~~

**朴素贝叶斯分类器：**

.. math::

   \begin{eqnarray}
   y_i=f(x_i) & & {} = \arg\underset{c_k}{\max} P(y_i=c_k|x_i) \nonumber \\
   	       & & {} = \arg\underset{c_k}{\max} P(y_i=c_k)\prod_{j}P(x_{ij}|y_i=c_k) \nonumber
   \end{eqnarray}

**估计 :math:`P(y_i=c_k)`\ ：**

:math:`P(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}`

**估计 :math:`P(x_{ij}|y_i=c_k)`\ ：**

假设第\ :math:`j`\ 个特征\ :math:`x_{ij}`\ 的取值集合为
:math:`\{a_{j1}, a_{j2}, \ldots, a_{jS_j}\}`

:math:`P(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j`

**算法：**

给定训练数据集 :math:`T = \{(x_i, y_i), i= 1, 2, \ldots, N\}`

其中：

-  :math:`x_i = (x_{i1}, x_{i2}, \ldots, x_{ip})`\ ；

-  :math:`x_{ij}`\ 是第 :math:`i` 个样本的第 :math:`j` 个 特征；

-  :math:`x_{ij}\in \{a_{j1}, a_{j2}, \ldots, a_{jS_j}\}`\ ，\ :math:`a_{jl}, l=1, 2, \ldots, S_j`\ 是第
   :math:`j` 个特征可能取的第 :math:`l` 个值；

-  :math:`y\in \{c_1, c_2, \ldots, c_k\}`

..

   1. 计算先验概率及条件概率
      :math:`P(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N}, i = 1, 2, \ldots, N, k = 1, 2, \ldots, K`
      :math:`P(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j`

   2. 对于给定的样本 :math:`(x_{i1}, x_{i2}, \ldots, x_{ip})`\ ，计算
      :math:`P(y_i=c_k)\prod_{j=1}^{p}P(x_{ij}=a_{jl}|y_i=c_k), k = 1, 2, \ldots, K`

   3. 确定样本 :math:`x_i` 的类
      :math:`y_i = \arg\underset{c_k}{\max}P(y_i=c_k)\prod_{j=1}^{p}P(x_{ij}=a_{jl}|y_i=c_k)`

.. _header-n66:

贝叶斯估计
~~~~~~~~~~

   极大似然估计可能会出现所要估计得概率值为0的情况，这时会影响到后验概率的计算结果，使分类产生偏差；

估计先验概率：

:math:`P_\lambda(y_i=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)+\lambda}{N+K\lambda}, i = 1, 2, \ldots, N, k = 1, 2, \ldots, K`

估计条件概率：

:math:`P_{\lambda}(x_{ij} = a_{jl}|y_i=c_k)=\frac{\sum_{i=1}^{N}I(x_{ij} = a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^{N}I(y_i=c_k)+S_j\lambda}, j= 1, 2, \ldots, p; l = 1, 2, \ldots, S_j;\lambda\geq 0`

-  当 :math:`\lambda = 0` 时，极大似然估计；

-  当 :math:`\lambda = 1` 时，拉普拉斯平滑(Laplace smoothing)，常取
   :math:`\lambda = 1`\ ；
