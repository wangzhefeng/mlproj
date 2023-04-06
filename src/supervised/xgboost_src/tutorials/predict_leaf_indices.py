# -*- coding: utf-8 -*-


# ***************************************************
# * File        : predict_leaf_indices.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-22
# * Version     : 0.1.072200
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from watchgod import watch

import xgboost as xgb

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XGBOOST_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(XGBOOST_ROOT_DIR, "demo")
train_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.train")
test_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.test")


# data
dtrain = xgb.DMatrix(train_dir)
dtest = xgb.DMatrix(test_dir)

# params
param = {
    "max_depth": 2,
    "eta": 1,
    "objective": "binary:logistic",
}

# model performance
watchlist = [
    (dtest, "eval"),
    (dtrain, "train"),
]

# model
bst = xgb.train(
    param,
    dtrain,
    num_boost_round = 3,
    watchlist = watchlist,
)

# model predict
print("start testing predict the leaf indices")
print("predict using first 2 tree")
leafindex = bst.predict(
    dtest,
    iteration_range = (0, 2),
    pred_leaf = True,
    strict_shape = True,
)
print(leafindex.shape)
print(leafindex)

print("predict all trees")
leafindex = bst.predict(
    dtest,
    pred_leaf = True,
)
print(leafindex.shape)




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

