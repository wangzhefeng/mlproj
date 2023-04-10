# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mclf_wine.py
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

import pandas as pd
from sklearn.datasets import load_wine


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_wine(logging: bool = False):
    wine = load_wine()
    wine_df = pd.DataFrame(
        data = wine.data,
        columns = wine.feature_names,
    )
    wine_df["WineType"] = wine.target

    if logging:
        print(wine_df.head())
        print(wine_df.shape)
    
    return wine




# 测试代码 main 函数
def main():
    wine_df = get_wine(True)

if __name__ == "__main__":
    main()
