# -*- coding: utf-8 -*-


# ***************************************************
# * File        : sklearn_demo_1.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-09
# * Version     : 0.1.080922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from get_data import get_lgb_train_test_data
from get_data import data_path
from get_data import model_saved_path
from custom_metrics import rmsle, rae


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


# data
train_path = os.path.join(data_path, "lgb_data/regression/regression.train")
test_path = os.path.join(data_path, "lgb_data/regression/regression.test")
X_train, y_train, X_test, y_test, lgb_train, lgb_eval = get_lgb_train_test_data(
    train_path, 
    test_path, 
    weight_path = []
)

# model training
gbm = lgb.LGBMRegressor(
    num_leaves = 31, 
    learning_rate = 0.05, 
    n_estimators = 20
)
gbm.fit(
    X_train, 
    y_train, 
    eval_set = [(X_test, y_test)], 
    eval_metric = "l1", 
    early_stopping_rounds = 5
)

# model predicting
y_pred = gbm.predict(
    X_test, 
    num_iteration = gbm.best_iteration_
)

# eval
lgb_sklearn_rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"The rmse of prediction is: {lgb_sklearn_rmse}")

# feature imoportances
print(f"Feature importances: {list(gbm.feature_importances_)}")




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
