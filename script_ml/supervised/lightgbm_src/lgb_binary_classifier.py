# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lgb_binary_classifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-09
# * Version     : 0.1.040922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

from datasets.bclf_cancer import get_cancer

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
breast_cancer = get_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    breast_cancer.data, 
    breast_cancer.target
)
print(f"\nTrain/Test Sizes : {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

# lgb data
train_dataset = lgb.Dataset(
    X_train, 
    y_train, 
    feature_name = breast_cancer.feature_names.tolist()
)
test_dataset = lgb.Dataset(
    X_test, 
    y_test, 
    feature_name = breast_cancer.feature_names.tolist()
)

# ------------------------------
# model
# ------------------------------
# model trianing
params = {
    "objective": "binary",
    "verbosity": -1,
}
booster = lgb.train(
    params,
    train_set = train_dataset,
    valid_sets = (test_dataset,),
    num_boost_round = 10
)

# model predict




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
