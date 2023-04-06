# -*- coding: utf-8 -*-


# ***************************************************
# * File        : generalized_linear_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-21
# * Version     : 0.1.072122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import xgboost as xgb


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None
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
    "objective": "binary:logistic",
    "booster": "gblinear",
    "alpha": 0.0001,
    "lambda": 1,
}

# model performance
watchlist = [
    (dtest, "eval"),
    (dtrain, "train"),
]

# model
num_round = 4
bst = xgb.train(
    param,
    dtrain,
    num_round,
    watchlist,
)

# model predict
preds = bst.predict(dtest)
labels = dtest.get_label()
print("error=%f" % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

