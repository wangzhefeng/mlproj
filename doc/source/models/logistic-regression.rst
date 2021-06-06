.. _header-n2:

Logistic Regression
===================

.. _header-n3:

角度1 从线性模型出发
--------------------

**线性模型：**

:math:`y=f(x)=\omega \cdot x + b, y\in R`

**二分类模型：**

:math:`y = f(x), y \in \{0, 1\}`

**伯努利分布：**

:math:`y \sim b(0, p)`

假设事件发生(\ :math:`y=1`)的概率为:

:math:`p = P(y = 1)`

那么事件不发生(\ :math:`y=0`)的概率为：

:math:`1-p = P(y = 0)`

**the odds of experiencing an event**\ ：

:math:`odds = \frac{p}{1-p}`

取对数：

:math:`log(odds)= \log\Big(\frac{p}{1-p}\Big)`

其中：

:math:`log(odds) \in [-\infty, +\infty]`

**线性模型：**

:math:`\log\Big(\frac{p}{1-p}\Big) = g(x) = \omega \cdot x + b, \log\Big(\frac{p}{1-p}\Big) \in [-\infty, +\infty]`

因此：

:math:`p = \frac{1}{1+e^{-g(x)}}`

**Logistic Regression 模型：**

:math:`y = f(x), y \in \{0, 1\}`

.. math::

   \left\{
   \begin{array}{ll}
   P(y=1|x) =  \sigma(x) \\
   P(y=0|x) = 1-\sigma(x)
   \end{array} 
   \right.

其中\ :math:`\sigma(x)为sigmoid函数`\ ：

:math:`\sigma(x) = \frac{1}{1+e^{-(\omega \cdot x + b)}}`

.. _header-n29:

Loss Function
-------------

**Logistic Regression 模型：**

:math:`y = f(x), y \in \{0, 1\}`

.. math::

   \left\{
   \begin{array}{ll}
   P(y=1|x) =  \sigma(x) \\
   P(y=0|x) = 1-\sigma(x)
   \end{array} 
   \right.

其中\ :math:`\sigma(x)为sigmoid函数`\ ：

:math:`\sigma(x) = \frac{1}{1+e^{-(\omega \cdot x + b)}}`

**极大似然估计思想：**

给定数据集：
:math:`\{(x_i, y_i)\}`\ ，其中：\ :math:`i = 1, 2, \ldots, N`\ ，\ :math:`x_i \in R^p`\ ，\ :math:`y_i \in \{0, 1\}`\ ；

似然函数为：

:math:`l=\prod_{i=1}^{N}[\sigma(x_i)]^{y_{i}}[1-\sigma{x_i}]^{1-y_i}`

则对数似然函数为：

.. math::

   \begin{eqnarray}
   L(\omega) & & {}= \log(l) \nonumber\\
   		  & & {}= \log\prod_{i=1}^{N}[\sigma(x_i)]^{y_i}[1-\sigma(x_i)]^{1-y_i} \nonumber\\
   		  & & {}= \sum_{i=1}^{N}\log[\sigma(x_i)]^{y_i}[1-\sigma(x_i)]^{1-y_i} \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[\log[\sigma(x_i)]^{y_i}+\log[1-\sigma(x_i)]^{1-y_i}] \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[y_i\log\sigma(x_i)+(1-y_i)\log[1-\sigma(x_i)]] \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[y_i\log\frac{\sigma(x_i)}{1-\sigma(x_i)}+log[1-\sigma(x_i)]] \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[y_i(\omega \cdot x_i)-\log(1+e^{\omega\cdot x_i})] \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[y_i\log P(Y=1|x)+(1-y_i)\log(1-P(Y=1|x))] \nonumber\\
   		  & & {}= \sum_{i=1}^{N}[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)] \nonumber
   \end{eqnarray}

**Loss Function：**

:math:`L(\omega) = - \sum_{i=1}^{N} [y_{i} \log \hat{y}_{i} + (1-y_{i}) \log(1- \hat{y}_{i})]`

.. _header-n44:

Loss Function优化方法：
-----------------------

-  梯度下降法

-  拟牛顿法

.. _header-n52:

Logistic Regression 实现
------------------------

模型类型：

-  binray classification

-  multiclass classification

   -  One-vs-Rest classification

-  Multinomial classification

模型形式：

-  Logistic Regression with L1正则化

   -  :math:`\min_{w, c} \|w\|_1 + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1)`

-  Logistic Regression with L2正则化

   -  :math:`\min_{w, c} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1)`

模型学习算法：

-  liblinear

   -  坐标下降算法(coorinate descent algorithm, CD)

   -  算法稳健

-  newton-cg

-  lbfgs

   -  近似于Broyden-Fletcher-Goldfarb-Shanno算法的优化算法，属于准牛顿方法

   -  适用于小数据集，高维数据集

-  sag

   -  随机平均梯度下降(Stochastic Average Gradient descent)

   -  适用于大数据集，高维数据集

-  saga

   -  sag算法的变体

   -  适用于大数据集，高维数据集

-  SGDClassifier with log loss

   -  适用于大数据集，高维数据集

Scikit API:

.. code:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.linear_model import LogisticRegressionCV
   from sklearn.linear_model import SGDClassifier 

LogisticRegression：

.. code:: python

   lr = LogisticRegression(penalty = "l2", 
   					    dual = False,
   					    tol = 0.0001,
   					    C = 1.0,
   					    fit_intercept = True,
   					    intercept_scaling = 1,
   					    class_weight = None,
   					    random_state = None, 
   					    solver = "warn",
   					    max_iter = 100,
   					    multi_class = "warn",
   					    verbose = 0,
   					    warm_start = False,
   					    n_jobs = None)
   # Method
   lr.fit()
   lr.predict()
   lr.predict_proba()
   lr.predict_log_proba()
   lr.decision_function()
   lr.density()
   lr.get_params()
   lr.set_params()
   lr.score()
   lr.sparsify()

   # Attributes
   lr.classes_
   lr.coef_
   lr.intercept_
   lr.n_iter_

-  多分类

   -  multi_class = "ovr"：使用one-vs-rest模式

   -  multi_class = "multinomial"：使用cross-entropy loss

      -  仅支持：\ ``solver in ["lbfgs", "sag", "newton-cg"]``

-  其他

   -  ``dual = True, penalty = "l2"``

   -  ``solver in ["newton-cg", "sag", "lbfgs"], penalty = "l2"``

   -  ``solver = "liblinear", penalty in ["l2", "l1"]``

SGDClassifier：

.. code:: python

   # 使用SGD算法训练的线性分类器：SVM, Logistic Regression
   sgdc_lr = SGDClassifier(loss = 'log',
   						penalty = "l2",
   						alpha = 0.0001,
   						l1_ratio = 0.15,
   						fit_intercept = True,
   						max_iter = None, 
   						tol = None,
   						shuffle = True,
   						verbose = 0,
   						epsilon = 0.1, 
   						n_jobs = None,
   						random_state = None,
   						learning_rate = "optimal",
   						eta0 = 0.0,
   						power_t = 0.5, 
   						early_stopping = False,
   						validation_fraction = 0.1,
   						n_iter_no_change = 5,
   						class_weight = None, 
   						warm_start = False,
   						aveage = False,
   						n_iter = None)


LogisticRegressionCV：

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from sklearn import datasets
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler

   # data
   digits = datasets.load_digits()
   X, y = digits.data, digits.target
   X = StandardScaler().fit_transform(X)
   y = (y > 4).astype(np.int)

   for i, C in enmerate((1, 0.1, 0.01)):
   	clf_l1_LR = LogisticRegression(C = C, penalty = "l1", )
