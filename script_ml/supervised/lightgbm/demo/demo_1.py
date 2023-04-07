# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo_1.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-09
# * Version     : 0.1.080923
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from utils.get_data import get_lgb_train_test_data
from utils.get_data import data_path
from utils.get_data import model_saved_path


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


# data
train_path = os.path.join(data_path, "lgb_data/regression/regression.train")
test_path = os.path.join(data_path, "lgb_data/regression/regression.test")
X_train, y_train, X_test, y_test, lgb_train, lgb_eval = get_lgb_train_test_data(
    train_path, 
    test_path, 
    weight_paths = []
)

# model parameter config
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": {"l2", "l1"},
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

# model training
gbm = lgb.train(
    params, 
    lgb_train, 
    num_boost_round = 20, 
    valid_sets = lgb_eval, 
    early_stopping_rounds = 5
)

# model saving
model_path = os.path.join(model_saved_path, "lightgbm_sample_example_model.txt")
gbm.save_model(model_path)

# model predicting
y_pred = gbm.predict(
    X_test, 
    num_iteration = gbm.best_iteration,
)

# eval
lgb_rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"The rmse of prediction is: {lgb_rmse}")




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

