# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from get_data import get_lgb_train_test_data
from get_data import data_path
from get_data import model_saved_path
from custom_metrics import rmsle, rae

"""
#TODO 1.特征工程
#TODO 2.模型参数选择(交叉验证、GridSearch)
#TODO 3.过拟合处理
"""

# data
train_path = os.path.join(data_path, "lgb_data/regression/regression.train")
test_path = os.path.join(data_path, "lgb_data/regression/regression.test")
X_train, y_train, X_test, y_test, lgb_train, lgb_eval = get_lgb_train_test_data(train_path, test_path, weight_path = [])

# model training
gbm = lgb.LGBMRegressor(num_leaves = 31, learning_rate = 0.05, n_estimators = 20)
gbm.fit(X_train, y_train, eval_set = [(X_test, y_test)], eval_metric = "l1", early_stopping_rounds = 5)

# model predicting
y_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration_)

# eval
lgb_sklearn_rmse = mean_squared_error(y_test, y_pred) ** 0.5
print()
print(f"The rmse of prediction is: {lgb_sklearn_rmse}")

# feature imoportances
print()
print(f"Feature importances: {list(gbm.feature_importances_)}")
