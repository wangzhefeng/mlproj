<<<<<<<< HEAD:data_provider/data_loader.py
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101716
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
========
# -*- coding: utf-8 -*-

# ***************************************************
# * File        : fil_example.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-19
# * Version     : 1.0.091900
# * Description : fil-profile run fil_example.py
# * Link        : https://pythonspeed.com/fil/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

def make_big_array():
    return np.zeros((1024, 1024, 50))


def make_two_arrays():
    arr1 = np.zeros((1024, 1024, 10))
    arr2 = np.zeros((1024, 1024, 10))
    return arr1, arr2




# 测试代码 main 函数
def main():
    arr1, arr2 = make_two_arrays()
    another_arr = make_big_array()

if __name__ == "__main__":
    main()
>>>>>>>> 6a4fe4a8f1ca2bb97e0748fe9b4b4310683e6dab:data_provider/fil_example.py
