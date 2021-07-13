.. _header-n2:

ARIMA
=====

.. _header-n3:

test 1
------

.. code:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import os
   import warnings
   warnings.filterwarnings("ignore")
   from datetime import datetime
   from datetime import timedelta
   from matplotlib.pyplot import rcParams
   rcParams["figure.figsize"] = 15, 6
   # 平稳性检验(AD检验)
   from statsmodels.tsa.stattools import adfuller
   # 模型分解
   from statsmodels.tsa.seasonal import seasonal_decompose
   # ARIMA 模型
   from statsmodels.tsa.arima_model import ARIMA
   from statsmodels.tsa.stattools import acf, pacf

.. _header-n6:

ADFuller 平稳性检验
~~~~~~~~~~~~~~~~~~~

.. code:: python

   def stationarity_test(ts):
       # rolling statistics
       rollmean = pd.Series.rolling(ts, window = 12).mean()
       rollstd = pd.Series.rolling(ts, window = 12).std()

       orig = plt.plot(ts, color = "blue", label = "Original")
       mean = plt.plot(rollmean, color = "red", label = "Rolling mean")
       std = plt.plot(rollstd, color = "black", label = "Rolling std")
       plt.legend(loc = "best")
       plt.title("Rolling mean & Standard Deviation")
       plt.show()

       # Dickey Fuller test
       print("Results of Dickey-Fuller Test:")
       dftest = adfuller(ts, autolag = "AIC")
       dfountput = pd.Series(dftest[0:4], 
                             index = ["Test Statistic", 
                                      "p-value", 
                                      "#lag used", 
                                      "Number of observation used"])
       for key, value in dftest[4].items():
           dfountput["Critical Value(%s)" % key] = value

.. _header-n8:

ACF 自相关函数, PACF 偏自相关函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def acf_pacf(data):
       lag_acf = acf(data, nlags = 20)
       lag_pacf = pacf(data, nlags = 20, method = "ols")

       plt.subplot(121)
       plt.plot(lag_acf)
       plt.axhline(y = 0, linestyle = "--", color = "gray")
       plt.axhline(y = - 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
       plt.axhline(y = 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
       plt.title("Autocorrelation Function")

       plt.subplot(122)
       plt.plot(lag_pacf)
       plt.axhline(y = 0, linestyle = "--", color = "gray")
       plt.axhline(y = - 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
       plt.axhline(y = 1.96 / np.sqrt(len(data)), linestyle = "", color = "gray")
       plt.title("Partial Autocorrelation Function")

       plt.tight_layout()

.. _header-n11:

ARIMA
~~~~~

.. code:: python

   def arima_performance(data, order1):
       model = ARIMA(data, order = order1)
       results_arima = model.fit(disp = -1)
       results_arima_value = results_arima.fittedvalues
       results_future = result_airma.forecast(7)
       return results_arima_value, results_future

.. code:: python

   def arima_plot(data, results_arima_value):
       plt.plot(data)
       plt.plot(results_arima_value, color = "red")
       plt.title("RSS: %.4f" % sum((results_arima_value) ** 2))

.. code:: python

   def add_season(ts_recover_trend, startdate):
       ts2_season = ts2_season
       values = []
       low_conf_values = []
