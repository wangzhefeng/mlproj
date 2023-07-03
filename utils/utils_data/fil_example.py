# -*- coding: utf-8 -*-

# ***************************************************
# * File        : fil_example.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042418
# * Description : fil-profile run fil_example.py
# * Link        : https://pythonspeed.com/fil/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
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
