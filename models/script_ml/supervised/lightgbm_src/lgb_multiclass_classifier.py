# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lgb_multiclass_classifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-10
# * Version     : 0.1.041011
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from dataset.mclf_wine import get_wine


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data
wine = get_wine()
# data split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)
print("Train/Test Sizes : ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

train_dataset = lgb.Dataset(
    X_train, 
    y_train, 
    feature_name = wine.feature_names,
)
test_dataset = lgb.Dataset(
    X_test,
    y_test,
    feature_name = wine.feature_names,
)

# ------------------------------
# model
# ------------------------------
# model training
params = {
    "objective": "multiclass",
    "num_class": 3,
    "verbosity": -1,
}
booster = lgb.train(
    params,
    train_set = train_dataset,
    valid_sets = (test_dataset,),
    num_boost_round = 10,
)

# model predict
train_preds = booster.predict(X_train)
print(train_preds)
train_preds = np.argmax(train_preds, axis = 1)
print(train_preds)
print(f"\nTrain Accuracy Score: {accuracy_score(y_train, train_preds)}")

test_preds = booster.predict(X_test)
test_preds = np.argmax(test_preds, axis = 1)
print(f"\nTest Accuracy Score: {accuracy_score(y_test, test_preds)}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()