# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo_2.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-08
# * Version     : 0.1.040822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# --------------------------------------------
# data
# --------------------------------------------
# 原始数据
df_train = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train", 
    header = None, 
    sep = "\t"
)
df_test = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test", 
    header = None, 
    sep = "\t"
)
W_train = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.train.weight", 
    header = None
)[0]
W_test = pd.read_csv(
    "https://cdn.coggle.club/LightGBM/examples/binary_classification/binary.test.weight", 
    header = None
)[0]
y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis = 1)
X_test = df_test.drop(0, axis = 1)
num_train, num_feature = X_train.shape
print(num_train)
print(num_feature)

# 创建适用于 LightGBM 的数据
lgb_train = lgb.Dataset(X_train, y_train, weight = W_train, free_raw_data = False)
lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train, weight = W_test, free_raw_data = False)

# --------------------------------------------
# model training
# --------------------------------------------
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}
# 生成特征名称
feature_name = ["feature_" + str(col) for col in range(num_feature)]
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    valid_sets = lgb_train,
    feature_name = feature_name,
    categorical_feature = [21]
)

# --------------------------------------------
# 模型保存与加载
# --------------------------------------------
gbm.save_model("model.txt")
print("Dumping model to JSON ...")
model_json = gbm.dump_model()
with open("model.json", "w+") as f:
    json.dump(model_json, f, indent = 4)

# --------------------------------------------
# 查看特征重要性
# --------------------------------------------
print("Feature names:", gbm.feature_name())
print("Feature importances:", list(gbm.feature_importance()))

# --------------------------------------------
# 训练
# --------------------------------------------
gbm = lgb.train(
    params, 
    lgb_train, 
    num_boost_round = 10, 
    init_model = "model.txt", 
    valid_sets = lgb_eval, 
)
print("Finished 10 - 20 rounds with model file ...")

# --------------------------------------------
# 动态调整模型超参数
# --------------------------------------------
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    learning_rates = lambda iter: 0.05 * (0.99 ** iter),
    valid_sets = lgb_eval
)
print("Finished 20 ~ 30 rounds with deacy learning rates...")


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    valid_sets = lgb_eval,
    callbacks = [lgb.reset_parameter(bagging_fraction = [0.7] * 5 + [0.6] * 5)]
)
print("Finised 30 ~ 40 rounds with changing bagging_fraction...")

# --------------------------------------------
# 自定义损失函数
# --------------------------------------------
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return "error", np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = 10,
    init_model = gbm,
    fobj = loglikelihood,
    feval = binary_error,
    valid_sets = lgb_eval
)
print("Finished 40 ~ 50 rounds with self-defined objective function and eval metric...")

# --------------------------------------------
# 调参方法
# --------------------------------------------
# ----------------------
# 人工调参
# ----------------------
"""
- 提高速度
    - Use bagging by setting bagging_fraction and bagging_freq
    - Use feature sub-sampling by setting feature_fraction
    - Use small max_bin
    - Use save_binary to speed up data loading in future learning
    - Use parallel learning, refer to Parallel Learning Guide
- 提高准确率
    - Use large max_bin (may be slower)
    - Use small learning_rate with large num_iterations
    - Use large num_leaves (may cause over-fitting)
    - Use bigger training data
    - Try dart
- 处理过拟合
    - Use small max_bin
    - Use small num_leaves
    - Use min_data_in_leaf and min_sum_hessian_in_leaf
    - Use bagging by set bagging_fraction and bagging_freq
    - Use feature sub-sampling by set feature_fraction
    - Use bigger training data
    - Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
    - Try max_depth to avoid growing deep tree
    - Try extra_trees
    - Try increasing path_smooth
"""
# ----------------------
# 网格搜索
# ----------------------
lg = lgb.LGBMClassifier(silent = False)
param_dist = {
    "max_depth": [4, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [300, 900, 1200],
    "n_estimators": [50, 100, 150]
}
grid_search = GridSearchCV(lg, n_jobs = -1, param_grid = param_dist, cv = 5, scoring = "roc_auc", verbose = 5)
grid_search.fit(X_train, y_train)

grid_search.best_estimator_
grid_search.best_score_

# ----------------------
# 贝叶斯优化
# ----------------------
def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
    params = {"metrics": "auc"}
    params["max_depth"] = int(max(max_depth, 1))
    params["learning_rate"] = np.clip(0, 1, learning_rate)
    params["num_leaves"] = int(max(num_leaves, 1))
    params["n_estimators"] = int(max(n_estimators, 1))
    cv_result = lgb.cv(
        params, 
        df_train, 
        nfold = 5, 
        seed = 0, 
        verbose_eval = 200, 
        stratified = False
    )
    return 1.0 * np.array(cv_result["auc-mean"]).max()


lgbBO = BayesianOptimization(
    lgb_eval, 
    {
        "max_depth": (4, 8),
        "learning_rate": (0.05, 0.2),
        "num_leaves": (20, 1500),
        "n_estimators": (5, 200)
    },
    random_state = 0
)
lgbBO.maximize(init_points = 5, n_iter = 50, acq = "ei")
print(lgbBO.max)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
