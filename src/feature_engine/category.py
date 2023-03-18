# -*- coding: utf-8 -*-


# ***************************************************
# * File        : category.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-11-12
# * Version     : 0.1.111217
# * Description : 类别特征特征工程函数
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CategoryFeatureEngine:

    def __init__(self) -> None:
        pass

    
    def ValueCounts(self):
        """

        Examples:
            # data
            >>> df = pd.DataFrame({
                    '区域' : ['西安', '太原', '西安', '太原', '郑州', '太原'], 
                    '10月份销售' : ['0.477468', '0.195046', '0.015964', '0.259654', '0.856412', '0.259644'],
                    '9月份销售' : ['0.347705', '0.151220', '0.895599', '0236547', '0.569841', '0.254784']
                })
            # feature engine
            >>> df_counts = df['区域'].value_counts().reset_index()
            >>> df_counts.columns = ['区域', '区域频度统计']
            >>> print(df_counts)
            >>> df = df.merge(df_counts, on = ['区域'], how = 'left')
            >>> print(df)
        """
        pass







# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

