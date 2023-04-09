# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lgb_regressor.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-09
# * Version     : 0.1.040921
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 50)
import matplotlib.pyplot as plt

import sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from datasets.reg_boston import get_boston

import warnings
warnings.filterwarnings("ignore")
print(f"lightgbm version: {lgb.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data split
boston = get_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
print(f"\nTrain/Test Sizes : {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
# lgb data
train_dataset = lgb.Dataset(X_train, y_train, feature_name = boston.feature_names.tolist())
test_dataset = lgb.Dataset(X_test, y_test, feature_name = boston.feature_names.tolist())

# ------------------------------
# model
# ------------------------------
# model training
params = {
    "objective": "regression",
}
booster = lgb.train(
    params,
    train_set = train_dataset,
    valid_sets = (test_dataset, ),
    num_boost_round = 10,
    # valid_names = None,
    # categorical_feature = None,
    # verbose_eval = None,
)
train_preds = booster.predict(X_train)
print(f"\nTrain R2 Score: {r2_score(y_train, train_preds)}")

# model predict
test_preds = booster.predict(X_test)
idxs = booster.predict(X_test, pred_leaf = True)
shap_vals = booster.predict(X_test, pred_contrib = True)

print(f"\nTest R2 Score: {r2_score(y_test, test_preds)}")
print(f"\nidxs Shape: {idxs.shape}, \nidxs={idxs}")
print(f"\nshap_vals Shape: {shap_vals.shape}, \nshap_vals={shap_vals}")
print(f"\nShap Values of 0th Sample: {shap_vals[0]}")
print(f"\nPrediction of 0th using SHAP Values: {shap_vals[0].sum()}")
print(f"\nActual Prediction of 0th Sample: {test_preds[0]}")


# feature importance
gain_imp = booster.feature_importance(importance_type = "gain")
print(f"\ngain importance={gain_imp}")
split_imp = booster.feature_importance(importance_type = "split")
print(f"\nsplit importance={split_imp}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
