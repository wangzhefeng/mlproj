# -*- coding: utf-8 -*-


# ***************************************************
# * File        : boston_housing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-09
# * Version     : 0.1.040921
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_boston_df(logging: bool = False):
    raw_df = pd.read_csv(
        "http://lib.stat.cmu.edu/datasets/boston", 
        sep = "\s+", 
        skiprows = 22, 
        header = None
    )
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", 
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    boston_df = pd.DataFrame(data = data, columns = column_names)
    boston_df["Price"] = target
    if logging:
        print(boston_df.head())
        print(boston_df.shape)

    return boston_df, data, target


def get_boston():
    boston = load_boston()

    return boston




# 测试代码 main 函数
def main():
    boston_df = get_boston(logging = True)


if __name__ == "__main__":
    main()
