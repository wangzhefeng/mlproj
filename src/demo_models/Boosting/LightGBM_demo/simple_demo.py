# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from get_data import get_lgb_train_test_data
from get_data import data_path
from get_data import model_saved_path

"""
#TODO 1.特征工程
#TODO 2.模型参数选择(交叉验证、GridSearch)
#TODO 3.过拟合处理
"""

# data
train_path = os.path.join(data_path, "lgb_data/regression/regression.train")
test_path = os.path.join(data_path, "lgb_data/regression/regression.test")
X_train, y_train, X_test, y_test, lgb_train, lgb_eval = get_lgb_train_test_data(train_path, test_path, weight_paths = [])

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
y_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration)

# eval
lgb_rmse = mean_squared_error(y_test, y_pred) ** 0.5
print()
print(f"The rmse of prediction is: {lgb_rmse}")
