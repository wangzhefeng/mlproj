.. _header-n0:

Time Series Forecast
====================

.. _header-n3:

内容
----

-  Time Series forecast 的步骤

-  Dickey-Fuller 检验 和 ARIMA 模型

-  理论概念、Python 实现

.. _header-n11:

1.时间序列的特别之处
--------------------

时间序列，顾名思义，是指在恒定时间间隔收集记录的数据点的集合。时间序列分析及建模是对这些数据进行分析来确定序列的长期趋势，以便预测未来或执行其他形式的分析，但是时间序列分析又不同于常规的回归分析：

-  随时间自相关相关的，观测值是独立的线性回归模型的基本假设不成立

-  随着趋势的增加或减小，大多数的时间序列数据呈现出某种形式的季节性趋势，即在特定时间范围内的特定变化

.. _header-n19:

2.在 Pandas 中加载和处理时间序列
--------------------------------

-  数据来自一个航空公司的乘客数时间序列数据 ``AirPassengers.csv``

-  Pandas 有专门的库来处理时间序列对象，特别是 ``datetime64['ns']`` 类

-  time

-  datetime

-  calendar

-  dateutil

-  numpy

-  pandas

-  pyts

-  tsfresh

导入工具库：

.. code:: python

   import numpy as np
   import pandas as pd
   from datetime import datetime
   import matplotlib.pylab as plt
   from matplotlib.pylab import rcParams
   rcParams["figure.figsize"] = 15, 6

读取数据：

.. code:: python

   dateparse = lambda dates: pd.to_datetime(dates, "%Y-%m")
   data = pd.read_csv("../data/AirPassengers/AirPassenger.csv",
   				   parse_dates = ["Month"],
   				   index_col = "Month",
   				   date_parser = dateparse)

   print(data.shape)
   print(data.head())
   print(data.index)

.. code:: python

   # 将数据转换为 pd.Series
   ts = data["#Passengers"]
   print(ts.head())

.. _header-n48:

3.如何检验时间序列的平稳性(Stationarity)
----------------------------------------

-  如果说时间序列是平稳的，是因为它有一些特殊的统计性质，比如：均值和方差对于时间来说是一个恒定的常数

   -  恒定的均值

   -  恒定的方差

   -  不依赖于时间的自协方差(autovariance)

-  几乎所有的时间序列模型都是在平稳性的假设下才能进行的，即认为如果时间序列随着时间的推移具有特定的行为，那么将来它很可能会遵循相同的行为，此外，与非平稳时间序列相比，平稳序列的理论更成熟且更容易实现。

.. code:: python

   plt.plot(ts);

根据时间序列图可以看出 #Passengers
变量总体上呈增长趋势，并伴有一些季节性变化，但是谁都并不能总是能通过视觉推断来得到这样的结论，因此，需要使用更加科学严格的统计方法检验时间序列的平稳性：

-  Plotting Rolling Statistics

   -  观察移动均值(moving average) 和 移动方差(moving variance)
      是否随时间变化

-  Dickey-Fuller Test

   -  原假设：时间序列时非平稳的

   -  检验的结果通过比较检验统计量(Test Statistic)和一个临界值(Critical
      Values)的大小决定

      -  如果测试统计量小于临界值，拒绝原假设，认为时间序列是平稳的

.. code:: python

   from statsmodels.tsa.stattools import adfuller

   def test_stationarity(timeseries):
   	# determing rolling statistics
   	rolmean = timeseries.rolling(window = 12).mean()
   	rolstd = timeseries.rolling(window = 12).std()

   	# plot rolling statistics
   	orig = plt.plot(timeseries, color = "blue", label = "Original")
   	mean = plt.plot(rolmean, color = "red", label = "Rolling Mean")
   	std = plt.plot(rolstd, color = "black", label = "Rolling Std")
   	plt.legend(loc = "best")
   	plt.title("Rolling Mean and Standard Deviation")
   	plt.show(block = False)

   	# perform Dickey-Fuller test
   	print("Results of Dickey-Fuller Test:")
   	dftest = adfuller(timeseries, autolag = "AIC")
   	dfoutput = pd.Series(dftest[0:4], 
   						 index = ["Test Statistic", 
   						 		  "p-value", 
   						 		  "#Lags Used", 
   						 		  "Number of Observations Used"])
   	for key, value in dftest[4].items():
   		dfoutput["Cirtical Value (%s)" % key] = value
   	print(dfoutput)

   test_stationarity(ts)

从 Rolling Statistic
的时序图上可以看到：标准差随时间的变化很小，均值明显随时间在增加，因此
``ts`` 不是一个平稳序列。并且，DF
检验统计量也大于临界值(所有,比较的带符号的值)，证明 ``ts``
的确不是一个平稳序列，需要对 ``ts`` 进行平稳化处理

.. _header-n83:

4.如何使得时间序列平稳
----------------------

**什么因素使得时间序列非平稳？**

1. Trend(趋势)

   -  随时间变化的均值

2. Seasonality(季节性)

   -  特定时间范围内的变化

**使时间序列平稳的方法：**

-  基本原则是模拟或估计时间序列中的趋势和季节性，并从时间序列中删除这两个因素的序列，从而使得时间序列平稳。变得平稳的时间序列可以用来进行预测

.. _header-n100:

4.1 估计并消除趋势
~~~~~~~~~~~~~~~~~~

-  消除时间序列趋势的一个有效方法是对原时间序列进行转换(transformation),对于有显著正向趋势的时间序列，进行转换来惩罚较大的数值

   -  对数转换

   -  平方根转换

   -  立方根转换

   -  ...

-  对于转换后的时间序列能够很容易观察到趋势的存在，但是由于噪音的存在，不是很直观。可以对时间序列进行估计或模拟这种趋势，然后将其从序列中删除：

   -  聚合(Aggretation)

      -  taking average for a time period like monthly/weekly average

   -  平滑(Smoothing)

      -  rolling averages

   -  多项式拟合(Polynomial Fitting)

      -  拟合一个回归模型

**(1) 对数转换**

.. code:: 

   ts_log = np.log(ts)
   plt.plot(ts_log)

**(2) 移动平均(Moving Average, MA)**

-  根据时间序列的频率取 k 个连续值的平均值，这里取过去一年的平均值：

.. code:: python

   moving_avg = ts_log.rolling(window = 12).mean()
   plt.plot(ts_log)
   plt.plot(moving_avg, color = "red")

上面时间序列图中的红色曲线代表的是 rolling
mean，代表了时间序列的趋势(trend)，现在从时间序列中移除这个序列：

.. code:: python

   ts_log_moving_avg_diff = ts_log - moving_avg
   ts_log_moving_avg_diff.head()

删除序列中的前11个 ``NaN`` 值，对时间序列重新进行平稳性检验:

.. code:: python

   ts_log_moving_avg_diff.dropna(inplace = True)
   test_stationarity(ts_log_moving_avg_diff)

移动平均处理后的时间序列看起来是一个比较平稳的序列，平稳性检验的检验统计量小于
临界值的 5%，因此，可以说有 95%
的把握认为移动平均后的序列是一个平稳时间序列。

**(3) 移动加权平均**

然而，这种特定方法的缺点是必须严格定义时间间隔(time-priods)，上面我们采取了年平均值，但是在更加复杂的情况下，如股票价格预测，很难决定使用哪个时间间隔数据。因此，我们需要采用\ **移动加权平均值**\ ，其中离当前值更近的值被赋予更高的权重，对于权重的分配，有很多技术，比较常用的是\ **指数加权移动平均值**\ ，其中权重被分配给具有衰减因子的所有先前值

.. code:: python

   expwighted_avg = ts.ewm(halflife = 12).mean()
   plt.plot(ts_log)
   plt.plot(expwighted_avg, color = "red")

现在重新从 ``ts_log`` 中删除趋势序列因素，并且检查时间序列的平稳性：

.. code:: python

   ts_log_ewm_diff = ts_log - expwighted_avg
   test_stationarity(ts_log_ewm_diff)

这里，指数加权移动平均处理后的时间序列的平均值和标准差的变化幅度更小，并且，平稳性检验的检验统计量小于
1% 的临界值，所以相比于移动平均，指数加权移动平均的处理效果更好

.. _header-n151:

4.2 消除趋势和季节性
~~~~~~~~~~~~~~~~~~~~

   4.1
   讨论的简单趋势消除技术在大多数情况下都不起作用，尤其是对于具有高季节性的时间序列数据，因此需要使用更加复杂的消除趋势和季节性因素的方法：

-  差分

   -  将时间序列与前一时刻的观察值进行相减(taking the differece with a
      particular time lag)，可以改善序列的平稳性

-  分解

   -  对趋势和季节性进行建模并将其从模型中移除

**(1) 一阶差分(Differencing)**

.. code:: python

   ts_log_diff = ts_log - ts_log.shift()
   plt.plot(ts_log_diff)

检查差分处理后的序列平稳性：

.. code:: python

   ts_log_diff.dropna(inplace = True)
   test_stationarity(ts_log_diff)

平稳性检验后的序列平稳性检验结果很好，但是可以继续进行二阶，三阶，甚至更高阶的差分处理，并验证处理后的结果，相信可以得到一个比较好的时间序列。

**(2) 分解(Decomposing)**

-  时间序列中的趋势和季节性通过建模的方式与序列分离

.. code:: python

   from statsmodels.tsa.seasonal import seasonal_decompose

   decomposition = seasonal_decompose(ts_log)
   trend = decomposition.trend
   seasonal = decomposition.seasonal
   residual = decomposition.resid

   plt.subplot(411)
   plt.plot(ts_log, lable = "Original")
   plt.legend(loc = "best")

   plt.subplot(412)
   plt.plot(trend, lable = "Trend")
   plt.legend(loc = "best")

   plt.subplot(413)
   plt.plot(seasonal, label = "Seasonality")
   plt.legend(loc = "best")

   plt.subplot(414)
   plt.plot(residual, label = "Residuals")
   plt.legend(loc = "best")

   plt.tight_layout()

对残差(residuals)建模、检验残差的平稳性：

.. code:: python

   ts_log_decompose = residual
   ts_log_decompose.dropna(inplace = True)
   test_stationarity(ts_log_decompose)

平稳性检验的结果表明，残差序列非常接近于平稳序列。

.. _header-n179:

5.Time Series Forecasting
-------------------------

在执行趋势和季节性估计后，可能存在两种情况：

1. 时间序列是一个严格的平稳序列，在序列值之间没有依赖性

   -  可以对残差进行建模为白噪声

2. 时间序列值有显著的前后依赖关系

   -  使用统计模型，比如：ARIMA 进行建模

Time Series Forecasting Models:

1. ARIMA(Auto-Regression Intergrated Moving Averages)

   -  预测变量相关的模型参数(p, d, q)

      -  p: Number of AR(自回归) terms

         -  AR 项是因变量的滞后项，比如：如果 :math:`p=5`\ ，则
            :math:`x(t)` 的预测值将是
            :math:`x(t-1), x(t-2), x(t-3), x(t-4), x(t-5), x(t-6)`

      -  q: Numver of MA(移动平均) terms

         -  lagged forecast errors in prediction equation

      -  d: 差异数量(Number of Differences):

         -  the number of nonseasonal differences

如何确定模型的参数？

1. 自相关函数(ACF)

   -  时间序列与其自身滞后版本之间的相关性度量

2. 局部自相关函数(PACF)

   -  在消除已经通过干预比较解释的变化后，时间序列与其自身滞后版本的相关性

.. code:: python

   # ACF 和 PACF图
   from statsmodels.tsa.stattools import acf, pacf

   lag_acf = acf(ts_log_diff, nlags = 20)
   lag_pacf = pacf(ts_log_diff, nlags = 20, method = "ols")

   # Plot ACF
   plt.subplot(121)
   plt.plot(lag_acf)
   plt.axhline(y = 0, linetype = "--", color = "gray")
   plt.axhline(y = -1.96 / np.sqrt(len(ts_log_diff)), linestyle = "--", color = "gray")
   plt.axhline(y = 1.96 / np.sqrt(len(ts_log_diff)), linestyle = "--", color = "gray")
   plt.title("Autocorrelation Function")


   # Plot PACF
   plt.subplot(122)
   plt.plot(lag_pacf)
   plt.axhline(y = 0, linestyle = "--", color = "gray")
   plt.axhline(y = -1.96 / np.sqrt(len(ts_log_diff)), linestyle = "--", color = "gray")
   plt.axhline(y = 1.96 / np.sqrt(len(ts_log_diff)), linestyle = "--", color = "gray")
   plt.title("Partial Autocorrelation Function")
   plt.tight_layout()

在 ACF 和 PACF 图中，纵轴值 0
上下两侧的虚线代表的是置信区间，利用这两个图可以确定 :math:`p` 和
:math:`q` 的值：

-  p: PACF 图中首次超过置信区间的滞后值，p=2

-  q: ACF 图中第一次超过置信区间的滞后值，q=2

**AR Model:**

.. code:: python

   from statsmodels.tsa.arima_model import ARIMA

   ar_model = ARIMA(ts_Log, order = (2, 1, 0))
   results_ar = ar_model.fit(disp = -1)
   plt.plot(ts_log_diff)
   plt.plot(results_ar.fittedvalues, color = "red")
   plt.title("RSS: %.4f" % sum(results_ar.fittedvalues - ts_log_diff) ** 2)

**MA Model:**

.. code:: python

   ma_model = ARIMA(ts_log, order = (0, 1, 2))
   results_ma = ma_model.fit(disp = -1)
   plt.plot(ts_log_diff)
   plt.plot(results_ma.fittedvalues, color = "red")
   plt.title("RSS: %.4f" % sum(results_ma.fittedvalues - ts_log_diff) ** 2)

**Combined Model:**

.. code:: python

   arima_model = ARIMA(ts_log, order = (2, 1, 2))
   results_arima = arima_model.fit(disp = -1)
   plt.plot(ts_log_diff)
   plt.plot(results_arima.fittedvalues, color = "red")
   plt.title("RSS: %.4f" % sum(results_arima.fittedvalues - ts_log_diff) ** 2)

..

   上面三个模型中，AR 和 MA 模型的残差(RSS)比较接近，ARIMA
   模型的残差(RSS)相比于前两个模型，明显比较小，所以 ARIMA
   模型的结果是当前最优的。

**Taking it back to original scale:**

1. 第一步：将预测模型的结果保存为 pd.Series

.. code:: python

   predictions_arima_diff = pd.Series(results_arima.fittedvalues, copy = True)
   predictions_arima_diff.head()

.. code:: python

   predictions_arima_diff_cumsum = predictions_arima_diff.cumsum()
   predictions_arima_diff_cumsum.head()

.. code:: python

   predictions_arima_log = pd.Series(ts_log.ix[0], index = ts_log.index)
   predictions_arima_log = predictions_arima_log.add(predictions_arima_diff_cumsum, fill_value = 0)
   predictions_arima_log.head()

取指数并与原始时间序列进行比较：

.. code:: python

   predictions_arima = np.exp(predictions_arima_log)
   plt.plot(ts)
   plt.plot(predictions_arima)
   plt.title("RMSE: %.4f" % np.sqrt(sum(predictions_arima - ts) ** 2) / len(ts))

.. _header-n260:

Knowledge Points
----------------

Pandas Standard moving window functions:

-  Standard moving window functions

   -  ts.rolling(window).count()

   -  ts.rolling(window).sum()

   -  ts.rolling(window).mean()

   -  ts.rolling(window).median()

   -  ts.rolling(window).var()

   -  ts.rolling(window).std()

   -  ts.rolling(window).min()

   -  ts.rolling(window).max()

   -  ts.rolling(window).corr()

   -  ts.rolling(window).cov()

   -  ts.rolling(window).skew()

   -  ts.rolling(window).kurt()

   -  ts.rolling(window).apply()

   -  ts.rolling(window).aggregate()

   -  ts.rolling(window).quantile()

   -  ts.window().mean()

   -  ts.window().sum()

-  Standard expanding window functions

   -  ts.expanding(window).count()

   -  ts.expanding(window).sum()

   -  ts.expanding(window).mean()

   -  ts.expanding(window).median()

   -  ts.expanding(window).var()

   -  ts.expanding(window).std()

   -  ts.expanding(window).min()

   -  ts.expanding(window).max()

   -  ts.expanding(window).corr()

   -  ts.expanding(window).cov()

   -  ts.expanding(window).skew()

   -  ts.expanding(window).kurt()

   -  ts.expanding(window).apply()

   -  ts.expanding(window).aggregate()

   -  ts.expanding(window).quantile()

-  Exponentially-weighted moving window functions

   -  ts.ewm().mean()

   -  ts.ewm().std()

   -  ts.ewm().var()

   -  ts.ewm().corr()

   -  ts.ewm().cov()

.. _header-n347:

Reference:
----------

-  `A comprehensive beginner’s guide to create a Time Series Forecast
   (with Codes in Python and
   R) <https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/>`__
