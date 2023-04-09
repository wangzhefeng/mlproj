# -*- coding: utf-8 -*-


# ***************************************************
# * File        : bclf_cancer.py
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
from sklearn.datasets import load_breast_cancer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_cancer(logging: bool = False):
    # sklearn object
    breast_cancer = load_breast_cancer()
    # df
    breast_cancer_df = pd.DataFrame(
        data = breast_cancer.data, 
        columns = breast_cancer.feature_names
    )
    breast_cancer_df["TumorType"] = breast_cancer.target
    if logging:
        print(breast_cancer_df.head())
        print(breast_cancer_df.shape)
    
    return breast_cancer




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
