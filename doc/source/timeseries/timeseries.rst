
timeseries
============

目标
--------------------------------------

   1. 形成一套数据处理流程文档: ``时间序列分析.doc``
   2. 形成一个汇报PPT: ``时间序列分析.pptx``
   3. 形成一个在线文档: ``timeseries-doc``
   4. 形成算法处理库: ``timeseries_preprocessing``
   5. 形成一个资料库 ``timeseries``

时间序列分析 Python 工具
--------------------------------------

   -  pandas

      -  Date & Time
      -  Index
      -  Missing value

         -  ``pandas.DataFrame.ffill()``
         -  ``pandas.DataFrame.bfill()``
         -  ``fillna()``
         -  ``pandas.DataFrame.iterpolate()``

      -  Window Functions:

         -  rolling window
         -  expanding window

      -  Resample/频率变换

         -  上采样

            -  ts.resample().ffill()
            -  ts.resample().bfill()
            -  ts.resample().pad()
            -  ts.resample().asfreq()
            -  ts.resample().interpolate()

         -  降采样

            -  ts.resample().
            -  ts.resample().apply()
            -  ts.resample().aggregate()
            -  ts.resample().transform()
            -  ts.resample().pipe()

      -  Difference

         -  pandas.DataFrame().diff()
         -  pandas.Series().diff()

      -  Move Average

         -  pandas.DataFrame().rolling()
         -  pandas.Series().rolling()

      -  Random Walk


时间序列数据预处理方法
----------------------

   - 平稳性检验

      - 平稳是说系统参数不随时间变化，可采用参数检验法（分组，组间差异不应太大）和非参数检验法（游程检验，游程应适中，即样本数据出现的顺序没有趋势）进行检验。包括：

         - 均值、方差是否为常数
         - 自相关函数是否只于时间间隔有关

   - 正态性检验

      - 模型噪声需要是高斯白噪声。可采用 P-P 图和 Q-Q 图直观上进行判断，也可以采用卡方拟合优度检验正态性（落入各组的频数与理想期望接近）

   - 独立性检验

      - 独立的白噪声序列的自相关系数是冲激函数，则序列的自相关系数需接近冲激函数才能认为是独立的

   - 周期性检验

      - 分析序列中的周期性或准周期序列(如季节、星期等), 可通过直观比较得到周期性的定性结果:
        
         - 周期序列的功率谱会产生尖峰
         - 自相关系数是连续振荡波形
         - 数据的概率密度直方图是下凸，非周期是上凸
   
   - 趋势项检验

      - 趋势其实就是非平稳，包括均值、方差等的趋势。对均值或方差的趋势检验可以采用分组计算，再结合游程检验的思路进行。
        如果有趋势，则需要先提取趋势项再去掉趋势项。对趋势项可以采取滤波器消除，而对于多项式趋势可以采用最小二乘法提取。
        对缓慢变化趋势可采用数据时间窗和最小二乘平滑进行处理

   - 异常值检测、处理
      
      - 数据采集过程中可能产生虚假数据，需对奇异数据进行检测和剔除。可采用人工剔除的方式，也可用线性外推的方式剔除异常点

   - 缺失值检测、处理
   - 时间序列平稳化

      - 差分运算
      - 移动平均
      - 指数平滑
      - 时间序列分解
      - 日历调增
      - 数学变换

   - 对时间序列数据建立机器学习模型的数据处理方法

      - 特征构建

         - 基本日期时间特征
         - Lagging 特征
         - Window 特征

      - 交叉验证数据分割

         - Train-Test split
         - Multiple Train-Test split
         - Walk-Forward Validation

   - 傅里叶变换
   - 小波分析
   - 滤波
   - 时间序列可视化


时间序列预测方法
----------------

   1. 时间序列基本规则法
   2. 线性回归
   3. 传统时间序列建模方法
   4. 时间序列分解
   5. Facebook prophet
   6. 应用机器学习进行时间序列分析
   7. 应用深度学习进行时间序列分析

时间序列的分解
----------------

时间序列分解的工作原理是将时间序列分为三个部分：季节性、趋势、随机噪声

   -  季节性

      -  在固定的时间段内重复的模式。

   -  趋势

      -  指标的基本趋势。

   -  随即噪声

      -  季节性和趋势系列被删除后原始时间序列的残差

根据季节性变化，分解方法可分为：

   -  加法分解

      -  时间序列 = 季节性 + 趋势 + 随机噪声

   -  乘法分解

      -  时间序列 = 季节性 \* 趋势 \* 随机噪声

时间序列性质检测:

   -  季节性检测

      -  傅里叶变换
      -  `利用傅里叶变换检测季节性 <https://anomaly.io/detect-seasonality-using-fourier-transform-r/>`__

   -  趋势检测

      -  移动平均

时间序列异常检测：

   -  移动中位数分解异常检测
   -  正态分布异常值检测


reference
-------------

   -  `R package forecast <https://cran.r-project.org/web/packages/forecast/>`__
   -  `从数据中提取季节性和趋势 <https://anomaly.io/seasonal-trend-decomposition-in-r/index.html>`__
   -  `正态分布异常值检测 <https://anomaly.io/anomaly-detection-normal-distribution/index.html>`__
   -  `季节性地调整时间序列 <https://anomaly.io/seasonally-adjustement-in-r/index.html>`__
   -  `检测相关时间序列中的异常 <https://anomaly.io/detect-anomalies-in-correlated-time-series/index.html>`__
   -  `用移动中位数分解检测异常 <https://anomaly.io/anomaly-detection-moving-median-decomposition/index.html>`__
