.. _header-n2:

移动平均光滑
============

-  移动平均(simple moving average, SMA)

-  加权移动平均(weighted moving average, WMA)

-  指数加权移动平均(exponential weighted moving average, EMA, EWMA)

.. _header-n10:

简单移动平均
------------

:math:`m_t = \frac{1}{k}\sum_{i=1}^{k}y_{t-i}`

.. _header-n12:

加权移动平均
------------

:math:`m_t = \sum_{i=1}^{k}\omega_{i}y_{t-i}`

.. _header-n14:

指数加权移动平均
----------------

:math:`m_{t} = \beta \times m_{t-1} + (1 - \beta) \times y_{t}`
