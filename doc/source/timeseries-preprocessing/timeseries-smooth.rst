.. _header-n0:

时间序列平滑
============

-  .diff()

-  .shift()

-  .rolling()

-  .expanding()

-  .ewm()

-  .pct_change()

.. _header-n17:

1.差分运算、变化率
------------------

.. _header-n18:

1.1 差分运算
~~~~~~~~~~~~

**:math:`p` 阶差分:**

相距一期的两个序列值至之间的减法运算称为 :math:`1` 阶差分运算；对
:math:`1` 阶差分后序列在进行一次 :math:`1` 阶差分运算称为 :math:`2`
阶差分；以此类推，对 :math:`p-1` 阶差分后序列在进行一次 :math:`1`
阶差分运算称为 :math:`p` 阶差分.

:math:`\Delta x_{t} = x_{t-1} - x_{t-1}`

:math:`\Delta^{2} x_{t} = \Delta x_{t} - \Delta x_{t-1}`

:math:`\Delta^{p} x_{t} = \Delta^{p-1} x_{t} - \Delta^{p-1} x_{t-1}`

**:math:`k` 步差分：**

相距 :math:`k` 期的两个序列值之间的减法运算称为 :math:`k` 步差分运算.

:math:`\Delta_{k}x_{t} = x_{t} - x_{t-k}`

**差分运算 API:**

-  pandas.Series.diff

-  pandas.DataFrame.diff

-  pandas.DataFrame.percent

-  pandas.DataFrame.shift

.. code:: python

   # 1 阶差分、1步差分
   pandas.DataFrame.diff(periods = 1, axis = 0)

   # 2 步差分
   pandas.DataFrame.diff(periods = 2, axis = 0)

   # k 步差分
   pandas.DataFrame.diff(periods = k, axis = 0)

   # -1 步差分
   pandas.DataFrame.diff(periods = -1, axis = 0)

.. _header-n39:

1.2 百分比变化率
~~~~~~~~~~~~~~~~

当前值与前一个值之间的百分比变化

.. code:: python

   DataFrame/Series.pct_change(periods = 1, 
   					 fill_method = 'pad', 
   					 limit = None, 
   					 freq = None, 
   					 **kwargs)

-  periods

-  fill_method

-  limit

-  freq

.. _header-n52:

2.移动索引
----------

.. code:: python

   pandas.DataFrame.shift(periods, freq, axis, fill_value)

.. _header-n54:

3.移动平均、指数平滑
--------------------

API:

-  Standard moving window functions

   -  ts.rolling(window, min_periods, center).count()

   -  ts.rolling(window, min\ *periods, center, win*\ type).sum()

      -  win_type

         -  boxcar

         -  triang

         -  blackman

         -  hamming

         -  bartlett

         -  parzen

         -  bohman

         -  blackmanharris

         -  nuttall

         -  barthann

         -  kaiser (needs beta)

         -  gaussian (needs std)

         -  general_gaussian (needs power, width)

         -  slepian (needs width)

         -  exponential (needs tau)

   -  ts.rolling(window, min\ *periods, center, win*\ type).mean()

      -  win_type

         -  boxcar

         -  triang

         -  blackman

         -  hamming

         -  bartlett

         -  parzen

         -  bohman

         -  blackmanharris

         -  nuttall

         -  barthann

         -  kaiser (needs beta)

         -  gaussian (needs std)

         -  general_gaussian (needs power, width)

         -  slepian (needs width)

         -  exponential (needs tau)

   -  ts.rolling(window, min_periods, center).median()

   -  ts.rolling(window, min_periods, center).var()

   -  ts.rolling(window, min_periods, center).std()

   -  ts.rolling(window, min_periods, center).min()

   -  ts.rolling(window, min_periods, center).max()

   -  ts.rolling(window, min_periods, center).corr()

   -  ts.rolling(window, min_periods, center).cov()

   -  ts.rolling(window, min_periods, center).skew()

   -  ts.rolling(window, min_periods, center).kurt()

   -  ts.rolling(window, min_periods, center).apply(func)

   -  ts.rolling(window, min_periods, center).aggregate()

   -  ts.rolling(window, min_periods, center).quantile()

   -  ts.window().mean()

   -  ts.window().sum()

-  Standard expanding window functions

   -  ts.expanding(window, min_periods, center).count()

   -  ts.expanding(window, min\ *periods, center, win*\ type).sum()

      -  win_type

         -  boxcar

         -  triang

         -  blackman

         -  hamming

         -  bartlett

         -  parzen

         -  bohman

         -  blackmanharris

         -  nuttall

         -  barthann

         -  kaiser (needs beta)

         -  gaussian (needs std)

         -  general_gaussian (needs power, width)

         -  slepian (needs width)

         -  exponential (needs tau)

   -  ts.expanding(window, min\ *periods, center, win*\ type).mean()

      -  win_type

         -  boxcar

         -  triang

         -  blackman

         -  hamming

         -  bartlett

         -  parzen

         -  bohman

         -  blackmanharris

         -  nuttall

         -  barthann

         -  kaiser (needs beta)

         -  gaussian (needs std)

         -  general_gaussian (needs power, width)

         -  slepian (needs width)

         -  exponential (needs tau)

   -  ts.expanding(window, min_periods, center).median()

   -  ts.expanding(window, min_periods, center).var()

   -  ts.expanding(window, min_periods, center).std()

   -  ts.expanding(window, min_periods, center).min()

   -  ts.expanding(window, min_periods, center).max()

   -  ts.expanding(window, min_periods, center).corr()

   -  ts.expanding(window, min_periods, center).cov()

   -  ts.expanding(window, min_periods, center).skew()

   -  ts.expanding(window, min_periods, center).kurt()

   -  ts.expanding(window, min_periods, center).apply(func)

   -  ts.expanding(window, min_periods, center).aggregate()

   -  ts.expanding(window, min_periods, center).quantile()

-  Exponentially-weighted moving window functions

   -  ts.ewm(window, min\ *periods, center, win*\ type).mean()

      -  win_type

         -  boxcar

         -  triang

         -  blackman

         -  hamming

         -  bartlett

         -  parzen

         -  bohman

         -  blackmanharris

         -  nuttall

         -  barthann

         -  kaiser (needs beta)

         -  gaussian (needs std)

         -  general_gaussian (needs power, width)

         -  slepian (needs width)

         -  exponential (needs tau)

   -  ts.ewm(window, min_periods, center).std()

   -  ts.ewm(window, min_periods, center).var()

   -  ts.ewm(window, min_periods, center).corr()

   -  ts.ewm(window, min_periods, center).cov()

.. _header-n311:

6.1 Rolling
~~~~~~~~~~~

.. code:: python

   s = pd.Series(np.random.randn(1000),
   			  index = pd.date_range("1/1/2000", periods = 1000))
   s = s.cumsum()

   r = s.rolling(window = 60)

.. _header-n315:

6.2 Expanding
~~~~~~~~~~~~~

.. _header-n317:

6.3 指数加权移动(Exponentially-weighted moving window, EWM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
