.. _header-n0:

Prophet 入门
================

.. _header-n3:

1.Prophet 介绍
--------------

   -  Prophet is a procedure for forecasting time series data based on
      an additive model where non-linear trends are fit with yearly,
      weekly, and daily seasonality, plus holiday effects. It works best
      with time series that have strong seasonal effects and serveal
      seasons of historical data. Prophet is robust to missing data and
      shifts in the trend, and typically handles outiles well.

   -  Prophet 是 Facebook Core Data Science team 开发的开源软件，包括对
      R 和 Python 的支持，可以在 CRAN 和 PyPI 上下载.

   -  Prophet 的优点：

      -  Accurate and fast

      -  Fully automatic

      -  Tunable forecasts

      -  Available in R or Python

.. _header-n22:

2.Prophet 安装
--------------

.. _header-n23:

2.1 R 中的安装
~~~~~~~~~~~~~~

.. code:: r

   # install prophet on CRAN
   install.packages("prophet", type = "source")

.. _header-n25:

2.2 Python 中的安装
~~~~~~~~~~~~~~~~~~~

(1) 安装 ``pystan``:

-  Windows

   -  compiler

      -  python

      -  C++ compiler

      -  PyStan

-  Linux

   -  compilers:

      -  gcc(gcc64 on Red Hat)

      -  g++(gcc64-c++ on Red Hat)

      -  build-essential

   -  Python development tools

      -  python-dev

      -  python3-dev

-  Anaconda

   -  ``conda install gcc``

.. code:: shell

   # Windows
   $ pip install pystan
   # or
   $ conda install pystan -c conda-forge

(2) 安装 ``prophet``:

.. code:: shell

   # install on PyPI
   $ pip install fbprophet

   # install using conda-forge
   $ conda install -c conda-forge fbprophet

(3) 安装交互式图形库

.. code:: shell

   $ pip install plotly

.. _header-n69:

3.Prophet 使用(Python)
----------------------

.. _header-n70:

3.1 Prophet 使用帮助
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from fbprophet import Prophet
   help(Prophet)
   help(Prophet.fit)

.. _header-n72:

3.2 Quick Start
~~~~~~~~~~~~~~~

**Example:**

-  Question:

   -  Wikipedia page for `Peyton
      Manning <https://en.wikipedia.org/wiki/Peyton_Manning>`__.

-  `data <https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv>`__

导入工具库：

.. code:: python

   import pandas as pd
   from fbprophet import Prophet

数据读入：

.. code:: python

   df = pd.read_csv("./data/example_wp_log_peyton_manning.csv")
   df.head()

建立模型：

.. code:: python

   m = Prophet()
   m.fit(df)

模型预测：

.. code:: python

   future = m.make_future_dataframe(periods = 365)
   future.head()
   future.tail()

.. code:: python

   forecast = m.predict(future)
   forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

forecast 可视化：

.. code:: python

   fig1 = m.plot(forecast)

forecast 组件可视化：

-  trend

-  yearly seasonlity

-  weekly seasonlity

-  holidays

.. code:: python

   fig2 = m.plot_components(forecast)

.. code:: python

   from fbprophet.plot import plot_plotly
   import plotly.offline as py
   py.init_notebook_mode()

   fig = plot_plotly(m, forecast)
   py.iplot(fig)

.. _header-n106:

4. Python APIs
--------------

-  APIs:

   -  ``Prophet``

      -  ``fit``

      -  ``predict``

-  Input:

   -  dataframe with two columns

      -  ``ds``

         -  datestamp: ``YYYY-MM-DD``

         -  timestamp: ``YYYY-MM-DD HH:MM:SS``

      -  ``y``

         -  numeric

         -  measurement of forecast
