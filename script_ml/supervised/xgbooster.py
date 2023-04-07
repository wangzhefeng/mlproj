# -*- coding: utf-8 -*-


# ***************************************************
# * File        : xgboostclassifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040511
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import math

import numpy as np
from sklearn.model_selection import GridSearchCV
sys.path.append("..")

import xgboost as xgb


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def logloss(y_true, y_pred):
    label2num = dict(
        (name, i) for i, name in enumerate(sorted(set(y_true)))
    )
    return -1 * sum(
        math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf 
        for y, label in zip(y_pred, y_true)
    ) / len(y_pred)


class XGBoostClassifier():

    def __init__(self, num_boost_round = 10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round = None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(
            X, 
            label = [self.label2num[label] for label in y]
        )
        self.clf = xgb.train(
            params = self.params, 
            dtrain = dtrain, 
            num_boost_round = num_boost_round
        )
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis = 1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep = True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')

        if 'objective' in params:
            del params['objective']

        self.params.update(params)
        return self




# 测试代码 main 函数
def main():
    # model
    clf = XGBoostClassifier(
        eval_metric = "auc",
        num_class = 2,
        nthread = 4,
        silent = 1,
    )
    # params
    parameters = {
        "num_boost_round": [100, 250, 500],
        "eta": [0.05, 0.1, 0.3],
        "max_depth": [6, 9, 12],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.9, 1.0],
    }
    # model training
    clf = GridSearchCV(clf, parameters, n_jobs = 1, cv = 2)
    X_train = [[1, 2], 
               [3, 4], 
               [2, 1], 
               [4, 3], 
               [1, 0], 
               [4, 5]]
    Y_train = ["a", "b", "a", "b", "a", "b"]
    # model fit
    clf.fit(X_train, Y_train)
    best_parameters, score, _ = max(clf.grid_scores_, key = lambda x: x[1])
    print(f"score: {score}")
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r".format(param_name, best_parameters[param_name]))
    # model predict
    Y_test = [[1, 1]]
    prediction = clf.predict(Y_test)
    print(f"predicted: {prediction}")

if __name__ == "__main__":
    main()
