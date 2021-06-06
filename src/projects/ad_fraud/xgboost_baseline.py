# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from datetime import timedelta, datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import logging


# ==============================================================
# 读取数据集
# ==============================================================
print("=" * 25)
print('read data')

path = "/Users/zfwang/data/mldata/aichallenge_2019_ad_fraud/data/"
df_test = pd.read_csv(path + 'round1_iflyad_anticheat_testdata_feature.txt', sep = '\t')
df_train = pd.read_csv(path + 'round1_iflyad_anticheat_traindata.txt', sep = '\t')

df_uni = pd.concat([df_train, df_test], ignore_index = True)
df_uni['label'] = df_uni['label'].fillna(-1).astype(int)

print("train data's shape:", df_train.shape)
print("test data's shape:", df_test.shape)
print("df_uni's shape": df_test.shape)


# ==============================================================
# 特征工程
# ==============================================================
# ----------------------------------
# 特征分类
# ----------------------------------
print("=" * 25)
print("features classification")

# 类别型特征
cat_cols = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype', 'ip',
            'reqrealip', 'city', 'province', 'adidmd5', 'imeimd5', 'idfamd5',
            'openudidmd5', 'macmd5', 'dvctype', 'model', 'make', 'ntt',
            'carrier', 'os', 'osv', 'orientation', 'lan', 'h', 'w', 'ppi']

# 无用特征
drop_cols = ['sid', 'label', 'nginxtime']

# ----------------------------------
# 缺失值特征填充
# ----------------------------------
print("=" * 25)
print('fill null')

for cat_col in cat_cols:
    if df_uni[cat_col].isnull().sum() > 0:
        df_uni[cat_col].fillna('null_value', inplace = True)

# ----------------------------------
# 生成计数特征
# ----------------------------------
print("=" * 25)
print("generate count features")

def gen_value_counts(data, col):
    print('value counts', col)
    df_tmp = pd.DataFrame(data[col].value_counts().reset_index())
    df_tmp.columns = [col, 'tmp']
    r = pd.merge(data, df_tmp, how = 'left', on = col)['tmp']
    return r.fillna(0)

value_counts_col = ['pkgname', 'adunitshowid', 'ip', 'reqrealip',
                    'adidmd5', 'imeimd5', 'idfamd5', 'macmd5']

for col in value_counts_col:
    df_uni['vc_' + col] = gen_value_counts(df_uni, col)

# ----------------------------------
# 特征转换(cut)
# ----------------------------------
print("=" * 25)
print('features transform of cut')

def cut_col(data, col_name, cut_list):
    print('cutting', col_name)

    def _trans(array):
        count = array['box_counts']
        for box in cut_list:
            if count <= box:
                return 'count_' + str(box)
        return array[col_name]

    df_counts = pd.DataFrame(data[col_name].value_counts())
    df_counts.columns = ['box_counts']
    df_counts[col_name] = df_counts.index
    df = pd.merge(data, df_counts, on=col_name, how='left')
    column = df.apply(_trans, axis=1)
    return column


cut_col_dict = {
    ('pkgname', 'ver', 'reqrealip', 'adidmd5',
     'imeimd5', 'openudidmd5', 'macmd5', 'model', 'make'): [3],
    ('ip',): [3, 5, 10],
}

for cut_cols, cut_list in cut_col_dict.items():
    for col in cut_cols:
        df_uni[col] = cut_col(df_uni, col, cut_list)

# ----------------------------------
# 日期时间 特征
# ----------------------------------
print("=" * 25)
print('feature time')

df_uni['datetime'] = pd.to_datetime(df_uni['nginxtime'] / 1000, unit='s') + timedelta(hours=8)
df_uni['hour'] = df_uni['datetime'].dt.hour
df_uni['day'] = df_uni['datetime'].dt.day - df_uni['datetime'].dt.day.min()

cat_cols += ['hour']
drop_cols += ['datetime', 'day']

# ----------------------------------
# 
# ----------------------------------
print("=" * 25)
print('post process')

for col in cat_cols:
    df_uni[col] = df_uni[col].map(dict(zip(df_uni[col].unique(), range(0, df_uni[col].nunique()))))

# ----------------------------------
# 分割
# ----------------------------------
print("=" * 25)
print("train, vaild, test data index")

all_train_index = (df_uni['day'] <= 6).values
train_index     = (df_uni['day'] <= 5).values
valid_index     = (df_uni['day'] == 6).values
test_index      = (df_uni['day'] == 7).values
train_label     = (df_uni['label']).values

# ----------------------------------
# 删除无用特征
# ----------------------------------
print("= " * 25)
print("drop unuseful features")

for col in drop_cols:
    if col in df_uni.columns:
        df_uni.drop([col], axis=1, inplace=True)

# ----------------------------------
# 类别特征One-Hot重编码、连续数值特征稀疏化
# ----------------------------------
print("=" * 25)
print("类别特征One-Hot重编码、连续数值特征稀疏化")

ohe = OneHotEncoder()
mtx_cat = ohe.fit_transform(df_uni[cat_cols])

num_cols = list(set(df_uni.columns).difference(set(cat_cols)))
mtx_num = sparse.csr_matrix(df_uni[num_cols].astype(float).values)

mtx_uni = sparse.hstack([mtx_num, mtx_cat])
mtx_uni = mtx_uni.tocsr()

# ----------------------------------
# 低方差特征删除
# ----------------------------------
print("=" * 25)
print("低方差特征删除")

def col_filter(mtx_train, y_train, mtx_test, func = chi2, percentile = 90):
    feature_select = SelectPercentile(func, percentile = percentile)
    feature_select.fit(mtx_train, y_train)
    mtx_train = feature_select.transform(mtx_train)
    mtx_test = feature_select.transform(mtx_test)
    return mtx_train, mtx_test

# ==============================================================
# 模型数据准备
# ==============================================================
print("=" * 25)
print("prepare model train, vaild and test data")

# ----------------------------------
# 所有的训练数据，测试数据
# ----------------------------------
all_train_x, test_x = col_filter(
    mtx_uni[all_train_index, :],
    train_label[all_train_index],
    mtx_uni[test_index, :]
)
all_train_y = train_label[all_train_index]

# ----------------------------------
# 拟合模型的训练数据
# ----------------------------------
train_x = all_train_x[train_index[:all_train_x.shape[0]], :]
train_y = train_label[train_index]

# ----------------------------------
# 评估模型、超参数调优的验证数据
# ----------------------------------
val_x = all_train_x[valid_index[:all_train_x.shape[0]], :]
val_y = train_label[valid_index]


# ==============================================================
# 训练模型
# ==============================================================
print("=" * 25)
print('training')
print("=" * 25)

# ----------------------------------
# 定义模型评估指标
# ----------------------------------
def lgb_confusion_matrix():
    pass

def lgb_precision_recall():
    pass

def lgb_f1(labels, preds):
    score = f1_score(labels, np.round(preds))
    return 'f1', score, True

# ----------------------------------
# 定义模型(模型调参)
# ----------------------------------
xgbc = XGBClassifier(random_seed = 2019,
                     n_jobs = -1,
                     objective = "binary",
                     learning_rate = 0.1,
                     n_estimators = 4000,
                     num_leaves = 64,
                     max_depth = -1,
                     min_child_samples = 20)

print('best score', lgb.best_score_)

# ==============================================================
# 使用全部的 train data 和 调好迭代轮数训练模型，并用 test data 做预测
# ==============================================================
print("=" * 25)
print('predicting')

lgb.n_estimators = lgb.best_iteration_
lgb.fit(all_train_x, all_train_y)
test_y = lgb.predict(test_x)


# ==============================================================
# 创建submission.csv文件
# ==============================================================
print("=" * 25)
print("submission file")
print("=" * 25)

df_sub = pd.concat([df_test['sid'], pd.Series(test_y)], axis = 1)
df_sub.columns = ['sid', 'label']
df_sub.to_csv('/Users/zfwang/project/mlproj/projects/move_ad_fraud/submission_file/submit-{}.csv' \
    .format(datetime.now().strftime('%m%d_%H%M%S')), sep = ',', index = False)


