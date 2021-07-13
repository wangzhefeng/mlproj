
Prophet
=======


1.Prophet 简介
----------------

Facebook 开源了一个时间序列预测的算法，叫做 **fbprophet**，它的官方网址与基本介绍来自于以下几个网站:

    1. Github: https://github.com/facebook/prophet

    2. 官方网址: https://facebook.github.io/prophet/

    3. 论文地址: https://peerj.com/preprints/3190/

从官网的介绍来看，Facebook 所提供的 prophet 算法不仅可以处理时间序列存在一些异常值的情况，也可以处理部分缺失值的情形，
还能够几乎全自动地预测时间序列未来的走势。

从论文上的描述来看，这个 prophet 算法是基于时间序列分解和机器学习的拟合来做的，其中在拟合模型的时候使用了 pyStan 这个开源工具，
因此能够在较快的时间内得到需要预测的结果。

除此之外，为了方便统计学家，机器学习从业者等人群的使用，prophet 同时提供了 R 语言和 Python 语言的接口。

从整体的介绍来看，如果是一般的商业分析或者数据分析的需求，都可以尝试使用这个开源算法来预测未来时间序列的走势。


2.Prophet 的算法原理
---------------------

2.1 Prophet 数据的输入和输出
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.1.1 数据输入、数据输出
^^^^^^^^^^^^^^^^^^^^^^^^


首先让我们来看一个常见的时间序列场景，黑色表示原始的时间序列离散点，深蓝色的线表示使用
时间序列来拟合所得到的取值，而浅蓝色的线表示时间序列的一个置信区间，也就是所谓的合理的
上界和下界。prophet 所做的事情就是：

    - (1)输入已知的时间序列的时间戳和相应的值；
    - (2)输入需要预测的时间序列的长度；
    - (3)输出未来的时间序列走势;
    - (4)输出结果可以提供必要的统计指标，包括拟合曲线，上界和下界等;


2.1.2 时间序列数据格式
^^^^^^^^^^^^^^^^^^^^^^^^

就一般情况而言，时间序列的离线存储格式为时间戳和值这种格式，更多的话可以提供时间序列的 ID，
标签等内容。因此，离线存储的时间序列通常都是以下的形式。

    ==================== ======== ======== ========
    date                 category value    label
    ==================== ======== ======== ========
    2017-10-20 22:00:00  id1      10       "unknow"
    2017-10-20 22:01:00  id1      9        "1"
    2017-10-20 22:02:00  id1      0        "0"
    ==================== ======== ======== ========

- 其中:

    - ``date`` 指的是具体的时间戳

    - ``category`` 指的是某条特定的时间序列 id

    - ``value`` 指的是在 date 下这个 category 时间序列的取值

    - ``label`` 指的是人工标记的标签

        - "0" 表示异常

        - "1" 表示正常

        - "unknown" 表示没有标记或者人工判断不清

2.1.3 fbprophet 所需要的时间序列格式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``fbprophet`` 所需要的时间序列也是这种格式的，根据官网的描述，只要用 ``csv`` 文件存储两列即可:
    
    - 第一列的名字是 ``ds``

        - 表示时间序列的时间戳
    
    - 第二列的名称是 ``y``
    
        - 表示时间序列的取值

通过 ``prophet`` 的计算，可以计算出如下的数值：

    - ``yhat``: 表示时间序列的预测值
    
    - ``yhat_lower``: 表示时间序列预测值的下界
    
    - ``yhat_upper``: 表示时间序列预测值的上界

两份表格如下面的两幅图表示:

    - 输入数据格式

        == ==================== ====
        id ds                   y
        == ==================== ====
        0  2017-10-20 22:00:00  9.5
        1  2017-10-20 22:01:00  8.5
        2  2017-10-20 22:02:00  8.1
        3  2017-10-20 22:02:00  8.0
        4  2017-10-20 22:02:00  7.8
        == ==================== ====

    - 输出数据格式

        ===== ==================== ======== ========== ============
        id    ds                   yhat     yhat_lowr  yhat_uppder
        ===== ==================== ======== ========== ============
        3265  2017-10-20 22:00:00  8.19     7.48       8.96
        3266  2017-10-20 22:01:00  8.52     7.79       9.26
        3267  2017-10-20 22:02:00  8.31     7.55       9.04
        3268  2017-10-20 22:02:00  8.14     7.42       8.86
        3269  2017-10-20 22:02:00  8.15     7.39       8.88
        ===== ==================== ======== ========== ============

2.2 Prophet 的算法实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在时间序列分析中，常见的分析方法叫做时间序列的分解(Decomposition of Time Series)，
时间序列分解的原理是将时间序列 :math:`y_{t}` 分为三个部分：

   -  季节性项 :math:`S_t`

      -  在固定的时间段内重复的模式

   -  趋势项 :math:`T_t`

      -  指标的基本趋势

   -  随机噪声项(剩余项) :math:`R_t`

      -  季节性和趋势项被删除后原始时间序列的残差

根据季节性项 :math:`S_t` 的变化，分解方法可分为两种：

    -  加法分解

        - :math:`y_{t} = S_t + T_t + R_t`

    -  乘法分解
    
        - :math:`y_{t} = S_t \times T_t \times R_t`，等价于 :math:`\ln y_{t} = \ln S_t + \ln T_t + \ln R_t`

        - 由于上面的等价关系，所以，有的时候在预测模型的时候，会先取对数，然后再进行时间序列的分解，就能得到乘法的形式。
          在 fbprophet 算法中，作者们基于这种方法进行了必要的改进和优化。

一般来说，在实际生活和生产环节中，除了季节项、趋势项、剩余项之外，通常还有节假日的效应。
所以，在 prophet 算法里面，作者同时考虑了以上四项，Prophet 算法就是通过拟合这几项，
然后最后把它们累加起来就得到了时间序列的预测值。也就是：

    - :math:`y(t) = g(t) + s(t) + h(t) + \epsilon_t`

    - 其中：

        - :math:`g(t)` 表示趋势项，它表示时间序列在非周期上面的变化趋势

        - :math:`s(t)` 表示周期项，或者称为季节项，一般来说是以周或者年为单位

        - :math:`h(t)` 表示节假日项，表示在当天是否存在节假日

        - :math:`\epsilon_t` 表示误差项或者称为剩余项

趋势项模型 :math:`g(t)` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 Prophet 算法里面，趋势项有两个重要的函数，

    - 一个是基于逻辑回归函数（logistic function）的
    
    - 另一个是基于分段线性函数（piecewise linear function）

首先，我们来介绍一下基于逻辑回归的趋势项是怎么做的。




.. code-block:: python

    m = Prophet(growth = "logistic")
    df["cap"] = 6
    m.fit(df)
    future = m.make_future_dataframe(periods = prediction_length, freq = "min")
    future["cap"] = 6





变点的选择(Changepoint Selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


对未来的预估(Trend Forecast Uncertainty)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

季节性趋势
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


节假日效应(Holidays and events)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




模型拟合(Model Fitting)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

时间序列已经可以通过增长项，季节项，节假日项来构建了:

    - :math:`y(t) = g(t) + s(t) + h(t) + \epsilon_t`

下一步我们只需要拟合函数就可以了，在 Prophet 里面，作者使用了 pyStan 这个开源工具中的 L-BFGS 
方法来进行函数的拟合。具体可以参考 forecast.py 里面的 stan_init 函数。


参数设置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 Prophet 中，用户一般可以设置以下四种参数：

    - Capacity：在增量函数是逻辑回归函数的时候，需要设置的容量值

    - Change Points：可以通过 ``n_changepoints`` 和 ``changepoint_range`` 来进行等距的变点设置，
      也可以通过人工设置的方式来指定时间序列的变点

    - 季节性和节假日：可以根据实际的业务需求来指定相应的节假日

    - 光滑参数：

        - :math:`\tau` = ``changepoint_prior_scale`` 可以用来控制趋势的灵活度

        - :math:`\sigma` = ``seasonality_prior_scale`` 用来控制季节项的灵活度

        - :math:`v` = ``holidays_prior_scale`` 用来控制节假日的灵活度

如果不想设置的话，使用 Prophet 默认的参数即可。



2.3 Prophet 的参数设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



3 Prophet 的实际使用
----------------------------------

3.1 Prophet 的简单使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.因为 Prophet 所需要的两列名称是 ‘ds’ 和 ‘y’，其中，’ds’ 表示时间戳，’y’ 表示时间序列的值，因此通常来说都需要修改 pd.dataframe 的列名字。如果原来的两列名字是 ‘timestamp’ 和 ‘value’ 的话，只需要这样写：

.. code-block:: python

    df = df.rename(columns = {
        "timestamp": "ds",
        "value": y
    })

2.如果 ``timestamp`` 是使用 unixtime 来记录的，需要修改成 ``YYYY-MM-DD hh:mm:ss`` 的形式：

.. code-block:: python

    df["ds"] = pd.to_datetime(df["ds"], unit = "s")

3.在一般情况下，时间序列需要进行归一化的操作，而 ``pd.DataFrame`` 的归一化操作也十分简单：

.. code-block:: python

    df["y"] = (df["y"] - df["y"].mean()) / (df["y"].std())

4.然后就可以初始化模型，然后拟合模型，并且进行时间序列的预测了

.. code-block:: python

    # 初始化模型
    model = Prophet()

    # 拟合模型
    model.fit(X_train)


.. code-block:: python

    # 计算预测值：periods 表示需要预测的点数，freq 表示时间序列的频率。
    future = model.make_future_dataframe(periods = 30, freq = "min")
    future.tail()
    forecast = model.predict(future)
    # 画出预测图
    model.plot(forecast)
    # 画出时间序列的分量
    model.plot_components(forecast)
    # 
    x1 = forecast['ds']
    y1 = forecast['yhat']
    y2 = forecast['yhat_lower']
    y3 = forecast['yhat_upper']
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.plot(x1, y3)
    plt.show()
    # rophet 预测的结果都放在了变量 forecast 里面
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())



.. code-block:: python

    X_test_forecast = model.predict(X_test)






3.2 Prophet 的参数设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prophet 的默认参数配置(源码 forecaster.py)：

.. code-block:: python

    def __init__(
        self,
        growth='linear',
        changepoints=None,
        n_changepoints=25, 
        changepoint_range=0.8,
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        holidays=None,
        seasonality_mode='additive',
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000):
        pass



增长函数
^^^^^^^^^^^^^^^

在 Prophet 里面，有两个增长函数，分别是分段线性函数（linear）和逻辑回归函数（logistic）。
而 ``m = Prophet()`` 默认使用的是分段线性函数（linear），并且如果要是用逻辑回归函数的时候，
需要设置 ``capacity`` 的值，i.e. ``df[‘cap’] = 100``，否则会出错。

.. code-block:: python

    m = Prophet()
    m = Prophet(growth = 'linear')
    m = Prophet(growth = 'logistic')


变点
^^^^^^^^^^^^^^^

在 Prophet 里面，变点默认的选择方法是前 80% 的点中等距选择 25 个点作为变点，
也可以通过以下方法来自行设置变点，甚至可以人为设置某些点。

.. code-block:: python

    m = Prophet(n_changepoints=25)
    m = Prophet(changepoint_range=0.8)
    m = Prophet(changepoint_prior_scale=0.05)
    m = Prophet(changepoints=['2014-01-01'])

变点的作图:

.. code-block:: python

    from fbprophet.plot import add_changepoints_to_plot

    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)


周期性
^^^^^^^^^^^^^^^

可以在 Prophet 中设置周期性，无论是按月(month)还是周(week)

- Monthly

    .. code-block:: python

        # Monthly
        model = Prophet(weekly_seasonality = False)
        model.add_seasonality(name = "monthly", period = 30.5, fourier_order = 5)

- Weekly

    .. code-block:: python

        # Weekly
        model = Prophet(weekly_seasonality = True)
        model.add_seasonality(name = "weekly", period = 7, fourier_order = 3, prior_scale = 0.1)



节假日
^^^^^^^^^^^^^^^

可以设置某些天是节假日，并且设置它的前后影响范围，也就是 ``lower_window`` 和 ``upper_window``

.. code-block:: python

    # build holidays datasets
    playoffs = pd.DataFrame({
        "holiday": "playoff",
        "ds": pd.to_datetime([
            '2008-01-13', '2009-01-03', '2010-01-16',
            '2010-01-24', '2010-02-07', '2011-01-08',
            '2013-01-12', '2014-01-12', '2014-01-19',
            '2014-02-02', '2015-01-11', '2016-01-17',
            '2016-01-24', '2016-02-07'
        ]),
        "lower_window": 0,
        "upper_window": 1,
    })

    superbowls = pd.DataFrame({
        "holiday": "superbowl",
        "ds": pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
        "lower_window": 0,
        "upper_window": 1,
    })
    holidays = pd.concat([playoffs, superbowl])

    # model 
    model = Prophet(holidays = holidays, holidays_prior_scale = 10.0)

.. note:: 

    对于商业分析等领域的时间序列，Prophet 可以进行很好的拟合和预测，但是对于一些周期性或者趋势性不是很强的时间序列，
    用 Prophet 可能就不合适了。但是，Prophet 提供了一种时序预测的方法，在用户不是很懂时间序列的前提下都可以使用这
    个工具得到一个能接受的结果。具体是否用 Prophet 则需要根据具体的时间序列来确定。


Reference
---------

-  `R-tutorial <https://otexts.com/fpp2/ts-objects.html>`__

-  `Blog <https://zr9558.com/2018/11/30/timeseriespredictionfbprophet/>`__


