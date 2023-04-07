# -*- coding: utf-8 -*-


# ***************************************************
# * File        : boston.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-03
# * Version     : 0.1.040319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
X, Y = datasets.load_boston(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 123)
X_train, X_test, Y_train, Y_test = torch.tensor(X_train, dtype = torch.float32), \
                                   torch.tensor(X_test, dtype = torch.float32), \
                                   torch.tensor(Y_train, dtype = torch.float32), \
                                   torch.tensor(Y_test, dtype = torch.float32)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
samples, features = X_train.shape
print(samples, features)


mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
