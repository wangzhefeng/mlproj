#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
import logging


# #################################################
# read data
# #################################################
print("read data")
path = "/home/zfwang/Downloads/data/mldata/aichallenge_2019_ad_fraud/data/"
df_train = pd.read_csv(path + "round1_iflyad_anticheat_traindata.txt", sep = "\t")
df_test = pd.read_csv(path + "round1_iflyad_anticheat_testdata_feature.txt", sep = "\t")
df_uni = pd.concat([df_train, df_test], ignore_index = True)
df_uni["label"] = df_uni["label"].fillna(-1).astype(int)


# #################################################
# feature engine
# #################################################
cat_cols = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype', 'ip',
           'reqrealip', 'city', 'province', 'adidmd5', 'imeimd5', 'idfamd5',
           'openudidmd5', 'macmd5', 'dvctype', 'model', 'make', 'ntt',
           'carrier', 'os', 'osv', 'orientation', 'lan', 'h', 'w', 'ppi']
drop_cols = ["sid", 'label', 'nginxtime']


# =================================================
# 对含有缺失值的特征用`null_value`进行填充
print("full null")
for cat_col in cat_cols:
    if df_uni[cat_col].isnull().sum() > 0:
        df_uni[cat_col].fillna("null_value", inplace = True)


# ================================================
# 生成特征
def gen_value_counts(data, col):
    print('value counts', col)
    df_tmp = pd.DataFrame(data[col].value_counts().reset_index())
    df_tmp.columns = [col, 'tmp']
    r = pd.merge(data, df_tmp, how = 'left', on = col)['tmp']
    return r.fillna(0)

value_counts_col = ["pkgname", "adunitshowid", "ip", "reqrealip",
                    "adidmd5", "imeimd5", "idfamd5", "macmd5"]

for col in value_counts_col:
    df_uni["vc_" + col] = gen_value_counts(df_uni, col)


# ================================================
print("cut")
def cut_col(data, col_name, cut_list):
    print("cutting", "col_name")
    
    def _trans(array):
        count = array(["box_counts"])
        for box in cut_list:
            if count <= box:
                return "count_" + str(box)
        return array[col_name]

    df_counts = pd.DataFrame(data[col_name].value_counts())
    df_counts.columns = ["box_counts"]
    df_counts[col_name] = df_counts.index
    df = pd.merge(data, df_counts, on = col_name, how = "left")
    column =  df.apply(_trans, axis = 1)
    return column

cut_col_dict = {
    ('pkgname', 'ver', 'reqrealip', 'adidmd5', 'imeimd5', 
     'openudidmd5', 'macmd5', 'model', 'make'): [3],
    ('ip',): [3, 5, 10],
}

for cut_cols, cut_list in cut_col_dict.items():
    for col in cut_cols:
        df_uni[col] = cut_col(df_uni, col, cut_list)


# ================================================
# 日期特征处理
print("feature time")
df_uni["datatime"] = pd.to_datetime(df_uni['nginxtime'] / 1000, unit = "s") + timedelta(hours = 8)
df_uni["hour"] = df_uni["datetime"].dt.hour
df_uni["day"] = df_uni["datetime"].dt.day - df_uni["datetime"].dt.day.min()
cat_cols += ["hour"]
drop_col += ["datetime", "day"]


# #################################################
# 
# #################################################
print("post process")
for col in cat_cols:
    df_uni[col] = df_uni[col].map(dict(
        zip(df_uni[col].unique(), range(0, df_uni[col].nunique()))
    ))

all_train_index = (df_uni["day"] <= 6).values
train_index = (df_uni["day"] <= 5).values
valid_index = (df_uni["day"] == 6).values
test_index = (df_uni["day"] == 7).values
train_label = (df_uni["label"]).values


# 删除无用特征
for col in drop_cols:
    if col in df_uni.columns:
        df_uni.drop([col], axis = 1, inplace = True)

# 类别型特征One-Hot encoding
ohe = OneHotEncoder()
mtx_cat = ohe.fit_transform(df_uni[cat_cols])

# 数值型特征转换为csr_matrix
num_cols = list(set(df_uni.columns).difference(set(cat_cols)))
mtx_num = sparse.csr_matrix(df_uni[num_cols].astype(float).values)

# 所有特征转换为csr_matrix
mtx_uni = sparse.hstack([mtx_num, mtx_cat])
mtx_uni = mtx_uni.tocsr()


# 特征选择--筛选方差高于阈值的特征
def col_filter(mtx_train, y_train, mtx_test, func = chi2, percentile = 90):
    feature_select = SelectPercentile(func, percentile = percentile)
    feature_select.fit(mtx_train, y_train)
    mtx_train = feature_select.transform(mtx_train)
    mtx_test = feature_select.transform(mtx_test)
    return mtx_train, mtx_test

all_train_x, test_x = col_filter(
    mtx_uni[all_train_index, :],
    train_label[all_train_index],
    mtx_uni[test_index, :]
)

# #################################################
# 模型数据准备
# #################################################
train_x = all_train_x[train_index[:all_train_x.shape[0]], :]
train_y = train_label[train_index]

val_x = all_train_x[valid_index[:all_train_x.shape[0]], :]
val_y = train_label[valid_index]


# #################################################
# 模型训练
# #################################################
print("train")

# metrics
def lgb_f1(labels, preds):
    score = f1_score(labels, np.round(preds))
    return "f1", score, True

# lightgbm
lgb = LGBMClassifier(random_seed = 2019,
                     n_jobs = -1,
                     objective = "binary",  # 二分类
                     learning_rage = 0.1,   # 学习率
                     n_estimators = 4000,   # 训练轮数
                     num_leaves = 64,       # 叶子节点数
                     max_depth = -1,        # 树最大深度
                     min_child_samples = 20,# 叶子节点最小样本数
                     min_child_weight = 9,  # 叶子节点最小权重值
                     subsample_freq = 1,    # 
                     subsample = 0.8,       # 
                     colsample_bytree = 0.8,# 每棵树的样本随机选择比例
                     reg_alpha = 1,         # alpha
                     reg_lambda = 5)        # lambda

# train lightgbm
lgb.fit(
    train_x,
    train_y,
    eval_set = [(train_x, train_y), (val_x, val_y)],
    eval_names = ["train", "val"],
    eval_metric = lgb_f1,
    early_stopping_rounds = 100,
    verbose = 10,
)

# train score
print("best score", lgb.best_score_)


# #################################################      
# 模型预测
# #################################################
print("predict")
all_train_y = train_label[all_train_index]
lgb.n_estimators = lgb.best_iteration_
lgb.fit(all_train_x, all_train_y)
test_y = lgb.predict(test_x)


# =================================================
print("generate submission file")
df_sub = pd.concat([df_test["sid"], pd.Series(test_y)], axis = 1)
df_sub.columns = ["sid", "label"]
df_sub.to_csv("submit-{}.csv".format(datetime.now().strftime("%m%d_%H%M%S")), 
              sep = ",", 
              index = False)