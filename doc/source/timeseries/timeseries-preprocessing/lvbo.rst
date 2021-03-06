.. _header-n0:

滤波
====

.. _header-n3:

1.滤波算法
----------

.. _header-n4:

1.1 限幅滤波
~~~~~~~~~~~~

-  方法：

   -  根据经验判断，确定两次采样允许的最大偏差值，假设为
      :math:`\delta`\ ，每次检测到新的值是判断：如果本次值与上次值之差小于等于
      :math:`\delta`\ ，则本次值有效；如果本次值与上次值之差大于
      :math:`\delta`\ ，则本次值无效，放弃本次值，用上一次值代替本次值；

-  优点：

   -  能有效克服因偶然因素引起的脉冲干扰；

-  缺点：

   -  无法抑制周期性的干扰，平滑度差

.. _header-n21:

1.2 中位数滤波
~~~~~~~~~~~~~~

-  方法：

   -  连续采样 :math:`N` 次（\ :math:`N` 取奇数），把 :math:`N`
      次采样值按照大小排列，取中间值为本次有效值；

-  优点：

   -  能有效克服因偶然因素引起的波动干扰，对温度、液位的变化缓慢的被测参数有良好的的滤波效果；

-  缺点：

   -  对流量、速度等快速变化的参数不适用；

.. _header-n38:

1.3 算法平均滤波
~~~~~~~~~~~~~~~~

-  方法：连续取 :math:`N` 个采样值进行算术平均运算，\ :math:`N`
   值较大时：信号平滑度较高，但灵活性较低，\ :math:`N`
   值较小时：信号平滑度较低，但灵敏度较高。\ :math:`N`
   值的选取：一般流量：\ :math:`N=12`\ ，压力：\ :math:`N = 4`\ ；

-  优点：适用于对一般具有随机干扰的信号进行滤波，这样的信号的特点是有一个平均值，信号在某一数值范围附近上下波动；

-  缺点：对于测量速度较慢或要求数据计算速度较快的实时控制不适用，比较浪费
   RAM；

.. _header-n46:

1.4 递推平均滤波(滑动平均滤波)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  方法：

   -  把连续取 :math:`N` 个采样值看成一个队列队列的长度固定为 :math:`N`
      每次采样到一个新数据放入队尾，并扔掉原来队首的一次数据.(先进先出原则)
      把队列中的 :math:`N` 个数据进行算术平均运算，就可获得新的滤波结果
      :math:`N` 值的选取：流量，\ :math:`N=1`
      压力：\ :math:`N=4`\ ；液面，\ :math:`N=4~12`\ ；温度，\ :math:`N=1~4`\ ；

-  优点：

   -  对周期性干扰有良好的抑制作用，平滑度高适用于高频振荡的系统；

-  缺点：

   -  灵敏度低对偶然出现的脉冲性干扰的抑制作用较差不易消除由于脉冲干扰所引起的采样值偏差不适用于脉冲干扰比较严重的场合比较浪费
      RAM；

.. _header-n63:

1.5 中位数平均滤波(防脉冲干扰平均滤波)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  方法：

   -  相当于\ ``中位值滤波法 + 算术平均滤波法`` 连续采样 :math:`N`
      个数据，去掉一个最大值和一个最小值然后计算 :math:`N-2`
      个数据的算术平均值N值的选取：\ :math:`3~14`\ ；

-  优点：

   -  融合了两种滤波法的优点对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差；

-  缺点：

   -  测量速度较慢，和算术平均滤波法一样比较浪费 RAM；

.. _header-n80:

1.6 限幅平均滤波
~~~~~~~~~~~~~~~~

-  方法：

   -  相当于
      ``限幅滤波法 + 递推平均滤波法``\ 每次采样到的新数据先进行限幅处理，再送入队列进行递推平均滤波处理；

-  优点：

   -  融合了两种滤波法的优点对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差；

-  缺点：

   -  比较浪费 RAM；

.. _header-n97:

1.7 一阶滞后滤波
~~~~~~~~~~~~~~~~

-  方法：

   -  取 :math:`a=0~1` 本次滤波结果 :math:`=(1-a)` 本次采样值+
      a*上次滤波结果；

-  优点：

   -  对周期性干扰具有良好的抑制作用 适用于波动频率较高的场合；

-  缺点：

   -  相位滞后，灵敏度低滞后程度取决于a值大小不能消除滤波频率高于采样频率的1/2的干扰信号；

.. _header-n114:

1.8 加权递推平均滤波
~~~~~~~~~~~~~~~~~~~~

-  方法：

   -  是对递推平均滤波法的改进，即不同时刻的数据加以不同的权通常是，越接近现时刻的数据，权取得越大。给予新采样值的权系数越大，则灵敏度越高，但信号平滑度越低；

-  优点：、

   -  适用于有较大纯滞后时间常数的对象 和采样周期较短的系统；

-  缺点：

   -  对于纯滞后时间常数较小，采样周期较长，变化缓慢的信号不能迅速反应系统当前所受干扰的严重程度，滤波效果差；

.. _header-n131:

1.9 消抖滤波
~~~~~~~~~~~~

-  方法：

   -  设置一个滤波计数器将每次采样值与当前有效值比较：如果采样值＝当前有效值，则计数器清零如果采样值<>当前有效值，则计数器+1，并判断计数器是否>=上限N(溢出)
      如果计数器溢出,则将本次值替换当前有效值,并清计数器；

-  优点：

   -  对于变化缓慢的被测参数有较好的滤波效果,
      可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动

-  缺点：

   -  对于快速变化的参数不宜如果在计数器溢出的那一次采样到的值恰好是干扰值,则会将干扰值当作有效值导入系统

.. _header-n148:

1.10 限幅滤波
~~~~~~~~~~~~~

-  方法：

   -  相当于 ``限幅滤波法 + 消抖滤波法`` 先限幅，后消抖；

-  优点：

   -  继承了 ``限幅`` 和 ``消抖`` 的优点改进了
      ``消抖滤波法``\ 中的某些缺陷，避免将干扰值导入系统；

-  缺点：

   -  对于快速变化的参数不宜；

.. _header-n166:

1.11 卡尔曼滤波
~~~~~~~~~~~~~~~

-  什么是卡尔曼滤波？

   -  你可以在任何含有不确定信息的动态系统中使用卡尔曼滤波，对系统下一步的走向做出有根据的预测，即使伴随着各种干扰，卡尔曼滤波总是能指出真实发生的情况；

   -  在连续变化的系统中使用卡尔曼滤波是非常理想的，它具有占内存小的优点（除了前一个状态量外，不需要保留其它历史数据），而且速度很快，很适合应用于实时问题和嵌入式系统；

-  算法的核心思想:

   -  根据当前的仪器 ``测量值`` 和上一刻的 ``预测值`` 和
      ``误差值``\ ，计算得到当前的最优量，再预测下一刻的量。

      -  核心思想比较突出的观点是把误差纳入计算，而且分为 ``预测误差``
         和 ``测量误差`` 两种，统称为 ``噪声``\ 。

      -  核心思想还有一个非常大的特点是：误差独立存在，始终不受测量数据的影响。

reference:

-  `1 <https://blog.csdn.net/u010720661/article/details/63253509>`__

-  `2 <http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/>`__

.. _header-n193:

2.滤波算法 Python 实现
----------------------

.. code:: python

   import scipy.sigbal as signal
   import numpy as np
   import pylab as lp
   import matplotlib.pyplot as plt
   import matplotlib

.. _header-n195:

2.1 算术平均滤波
~~~~~~~~~~~~~~~~

.. code:: python

   def ArithmeticAverage(inputs, per):
   	if np.shape(inputs)[0] % per != 0:
   		length = np.shape(inputs)[0] / per
   		for x in range(int(np.shape(inputs)[0]), int(length + 1) * per):
   			inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
   	inputs = inputs.reshape((-1, per))
   	mean = []
   	for tmp in inputs:
   		mean.append(tmp.mean())

   	return mean

.. _header-n197:

2.2 
~~~~
