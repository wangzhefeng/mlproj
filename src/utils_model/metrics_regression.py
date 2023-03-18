# -*- coding: utf-8 -*-


# ***************************************************
# * File        : metrics_regression.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-19
# * Version     : 0.1.031901
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from sklearn.metrics import make_scorer
from sklearn.metrics import explained_variance_score
# MAE
from sklearn.metrics import mean_absolute_error
# MSE
from sklearn.metrics import mean_squared_error
# MSLE
from sklearn.metrics import mean_squared_log_error
# MAE
from sklearn.metrics import median_absolute_error
# R2
from sklearn.metrics import r2_score


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def scoring():
	scoring_regressioner = {
		'R2': r2_score,
		'MES': mean_squared_error,
	}

	return scoring_regressioner





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
