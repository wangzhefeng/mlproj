# -*- coding: utf-8 -*-


# ***************************************************
# * File        : get_data.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-09
# * Version     : 0.1.080922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import numpy as np
import pandas as pd
import lightgbm as lgb


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


# TODO
def get_lgb_train_test_data(train_path, test_path, weight_paths = []):
    """
    读取LightGBM example demo 数据

    Args:
        train_path ([type]): [description]
        test_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    # read data
    df_train = pd.read_csv(train_path, header = None, sep = "\t")
    df_test = pd.read_csv(test_path, header = None, sep = "\t")
    print(df_train.head())
    print(df_test.head())

    # split data
    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis = 1)
    X_test = df_test.drop(0, axis = 1)

    # weight data
    if weight_paths != []:
        W_train = pd.read_csv(weight_paths[0], header = None)[0]
        W_test = pd.read_csv(weight_paths[1], header = None)[0]
        # lightgbm Dataset
        lgb_train = lgb.Dataset(X_train, y_train, weight = W_train, free_raw_data = False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train, weight = W_test, free_raw_data = False)
        return W_train, W_test, X_train, y_train, X_test, y_test, lgb_train, lgb_eval
    else:
        # lightgbm Dataset
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train)
        return X_train, y_train, X_test, y_test, lgb_train, lgb_eval




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

