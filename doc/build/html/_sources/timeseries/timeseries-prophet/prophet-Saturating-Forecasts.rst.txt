.. _header-n0:

Staturating Forecast
====================

.. _header-n3:

饱和预测(Staturating Forecast)
------------------------------

.. _header-n4:

Forecasting Growth
~~~~~~~~~~~~~~~~~~

-  By default, Prophet uses a **linear model** for its forecast. When
   **forecasting growth**, there is usually some maximum achievable
   point: total market size, total population size, etc. This is called
   the **carrying capacity**, and the forecast should **saturate** at
   this point.

-  Prophet allows you to make forecasts using a **logistic growth trend
   model**, with a specified carrying capacity.

.. code:: python

   import pandas as pd

   # ============================================
   # data
   # ============================================
   df = pd.read_csv("./data/example_wp_log_R.csv")
   # 设定一个 carrying capacity,根据数据或专家经验说明市场规模
   df["cap"] = 8.5

   # ============================================
   # model 1 - staturating maximum
   # ============================================
   m = Prophet(growth = "logistic")
   m.fit(df)

   future_df = m.make_future_dataframe(periods = 1826)
   future_df["cap"] = 8.5
   forecast = m.predict(future_df)
   fig = m.plot(forecast)

.. _header-n11:

Staturating Minimum
~~~~~~~~~~~~~~~~~~~

-  Staturating Capacity

   -  staturating maximum

   -  staturating minimum(maximum必须设定)

.. code:: python

   # data
   df = pd.read_csv("./data/example_wp_log_R.csv")
   df["y"] = 10 - df["y"]
   df["cap"] = 6
   df["floor"] = 1.5
   future["cap"] = 6
   future["floor"] = 1.5

   # ============================================
   # model 1 - staturating maximum and minimum
   # ============================================
   m = Prophet(growth = "logistic")
   m.fit(df)

   future_df = m.make_future_dataframe(periods = 1826)
   forecast = m.predict(future_df)
   fig = m.plot(forecast)
