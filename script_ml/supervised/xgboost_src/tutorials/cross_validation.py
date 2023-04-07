# -*- coding: utf-8 -*-


# ***************************************************
# * File        : cross_validation.py
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

import numpy as np
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

# params
param = {
    "max_depth": 2,
    "eta": 1,
    "objective": "binary:logistic",
}

# model
print("running cross validation")
xgb.cv(
    param,
    dtrain,
    num_boost_round = 2,
    nfold = 5,
    metrics = {"error"},
    seed = 0,
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv = True),
    ],
)


print("running cross validaion, disable standard deviation display")
res = xgb.cv(
    param,
    dtrain,
    num_boost_round = 10,
    nfold = 5,
    metrics = {"error"},
    seed = 0,
    callbacks = [
        xgb.callback.EvaluationMonitor(show_stdv = False),
        xgb.callback.EarlyStopping(3),
    ],
)
print(res)


print("running cross validation, with preprocessing function")
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.num(label == 0)) / np.sum(label == 1)
    param["scale_pos_weight"] = ratio
    return (dtrain, dtest, param)

xgb.cv(
    param,
    dtrain,
    num_boost_round = 2,
    nfold = 5,
    metrics = {"auc"},
    seed = 0,
    fpreproc = fpreproc,
)


print("running cross validation, with customized loss function")
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return "error", float(sum(labels != (preds > 0.0))) / len(labels)

param = {
    "max_depth": 2,
    "eta": 1,
}

xgb.cv(
    param,
    dtrain,
    num_boost_round = 2,
    nfold = 5,
    seed = 0,
    obj = logregobj,
    feval = evalerror,
)




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

