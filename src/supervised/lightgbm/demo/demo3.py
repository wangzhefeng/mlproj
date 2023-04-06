# -*- coding: utf-8 -*-


# ***************************************************
# * File        : demo3.py
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

import datetime
from unittest import result
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(info + "...\n\n")


# ==================================
# 读取数据
# ==================================
printlog("step1: reading data...")

breast = datasets.load_breast_cancer()

df = pd.DataFrame(
    breast.data, 
    columns = [x.replace(" ", "_") for x in breast.feature_names]
)
df["label"] = breast.target
df["mean_radius"] = df["mean_radius"].apply(lambda x: int(x))
df["mean_texture"] = df["mean_texture"].apply(lambda x: int(x))

dftrain, dftest = train_test_split(df)

categorical_features = [
    "mean_radius", 
    "mean_texture",
]

lgb_train = lgb.Dataset(
    dftrain.drop(["label"], axis = 1),
    label = dftrain["label"],
    categorical_feature = categorical_features,
)
lgb_valid = lgb.Dataset(
    dftest.drop(["label"], axis = 1),
    label = dftest["label"],
    categorical_feature = categorical_features,
    reference = lgb_train,
)


# ==================================
# 设置参数
# ==================================
printlog("step2: setting parameters...")

boost_round = 50
early_stop_rounds = 10
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": ["auc"],
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}


# ==================================
# 训练模型
# ==================================
printlog("step3: training model...")

results = {}
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round = boost_round,
    valid_sets = (lgb_valid, lgb_train),
    valid_names = ("validate", "train"),
    early_stopping_rounds = early_stop_rounds,
    evals_result = results,
)


# ==================================
# 评估模型
# ==================================
printlog("step4: evaluating model...")

y_pred_train = gbm.predict(
    dftrain.drop(["label"], axis = 1), 
    num_iteration = gbm.best_iteration,
)
y_pred_test = gbm.predict(
    dftest.drop(["label"], axis = 1),
    num_iteration = gbm.best_iteration,
)
print('train accuracy: {:.5} '.format(accuracy_score(dftrain['label'],y_pred_train>0.5)))
print('valid accuracy: {:.5} \n'.format(accuracy_score(dftest['label'],y_pred_test>0.5)))

lgb.plot_metric(results)
lgb.plot_importance(gbm, importance_type = "gain")


# ==================================
# 保存模型
# ==================================
printlog("step5: saving model...")

model_dir = "data/gbm.model"
print("model_dir: %s" % model_dir)
gbm.save_model(model_dir)



printlog("task end...")




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

