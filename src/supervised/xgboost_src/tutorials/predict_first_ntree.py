# -*- coding: utf-8 -*-


# ***************************************************
# * File        : predict_first_ntree.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-20
# * Version     : 0.1.072023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_svmlight_file


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XGBOOST_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(XGBOOST_ROOT_DIR, "demo")
train_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.train")
test_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.test")


def native_interface():
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
    num_round = 3
    bst = xgb.train(
        param,
        dtrain,
        num_round,
        watchlist,
    )

    print("start testing prediction from first n trees")
    # predict using first 1 tree
    label = dtest.get_label()
    ypred1 = bst.predict(dtest, iteration_range = (0, 1))
    # predict using all trees
    ypred2 = bst.predict(dtest)
    print("error of ypred1=%f" % (np.sum((ypred1 > 0.5) != label) / float(len(label))))
    print("error of ypred2=%f" % (np.sum((ypred2 > 0.5) != label) / float(len(label))))


def sklearn_interface():
    # data
    X_train, y_train = load_svmlight_file(train_dir)
    X_test, y_test = load_svmlight_file(test_dir)
    # model
    clf = xgb.XGBClassifier(n_estimators  = 3, max_depth = 2, eta = 1, use_label_encoder = False)
    clf.fit(X_train, y_train, eval_set = [(X_test, y_test)])
    assert clf.n_classes_ == 2

    print("start testing prediction from first n trees")
    # predict using first 1 tree
    ypred1 = clf.predict(X_test, iteration_range = (0, 1))
    # predict using all trees
    ypred2 = clf.predict(X_test)

    print("error of ypred1=%f" % (np.sum((ypred1 > 0.5) != y_test) / float(len(y_test))))
    print("error of ypred2=%f" % (np.sum((ypred2 > 0.5) != y_test) / float(len(y_test))))




# 测试代码 main 函数
def main():
    native_interface()
    sklearn_interface()


if __name__ == "__main__":
    main()

