# -*- coding: utf-8 -*-


# ***************************************************
# * File        : tuning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# ==========================================
# data
# ==========================================
def data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "Disbursed"
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    return train, test, predictors, target


# ==========================================
# XGBoost model and cross-validation
# ==========================================
def modelFit(alg, dtrain, predictors, target,
            scoring = 'auc', useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_train = xgb.DMatrix(data = dtrain[predictors].values, label = dtrain[target].values)
        cv_result = xgb.cv(params = xgb_param,
                            dtrain = xgb_train,
                            num_boost_round = alg.get_params()['n_estimators'],
                            nfold = cv_folds,
                            stratified = False,
                            metrics = scoring,
                            early_stopping_rounds = early_stopping_rounds,
                            show_stdv = False)
        alg.set_params(n_estimators = cv_result.shape[0])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric = scoring)
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    print("\nModel Report:")
    print("Accuracy: %.4f" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = "Feature Importances")
    plt.ylabel("Feature Importance Score")


# ==========================================
# parameter tuning
# ==========================================
def grid_search(train, predictors, target, param_xgb, param_grid, scoring, n_jobs, cv_method):
    grid_search = GridSearchCV(estimator = XGBClassifier(**param_xgb),
                                param_grid = param_grid,
                                scoring = scoring,
                                n_jobs = n_jobs,
                                iid = False,
                                cv = cv_method)
    grid_search.fit(train[predictors], train[target])
    print(grid_search.cv_results_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return grid_search


# -----------------------------------
# data
# -----------------------------------
train_path = "./data/GBM_XGBoost_data/Train_nyOWmfK.csv"
test_path = "./data/GBM_XGBoost_data/Test_bCtAN1w.csv"
train, test, predictors, target = data(train_path, test_path)


# -----------------------------------
# XGBoost 基于默认的learning rate 调节树的数量
# n_estimators
# -----------------------------------
param_xgb1 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model1 = XGBClassifier(**param_xgb1)

modelFit(alg = xgb_model1,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)

# -----------------------------------
# 调节基于树的模型
# max_depth, min_child_weight
# -----------------------------------
param_xgb_tree1 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree1,
            param_grid = param_grid_tree1,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)


# -----------------------------------
param_xgb_tree2 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 5,
    'min_child_weight': 2,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree2 = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree2,
            param_grid = param_grid_tree2,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb_tree3 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 2,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree3 = {
    'min_child_weight': [6, 8, 10, 12]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_xgb_tree1 = grid_search(train = train,
                            predictors = predictors,
                            target = target,
                            param_xgb = param_xgb_tree3,
                            param_grid = param_grid_tree3,
                            scoring = scoring,
                            n_jobs = n_jobs,
                            cv_method = cv_method)


scoring = "auc"
cv_method = 5
early_stopping_rounds = 50

modelFit(alg = grid_xgb_tree1.best_estimator_,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)


# -----------------------------------
# 调节基于树的模型
# gamma
# -----------------------------------
param_xgb_tree4 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree4 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree4,
            param_grid = param_grid_tree4,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb2 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model2 = XGBClassifier(**param_xgb2)

modelFit(alg = xgb_model2,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)




# -----------------------------------
# 调节基于树的模型
# subsample, colsample_bytree
# -----------------------------------
param_xgb_tree5 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree5 = {
    'subsample': [i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree5,
            param_grid = param_grid_tree5,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

# -----------------------------------
param_xgb_tree6 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_tree6 = {
    'subsample': [i/100.0 for i in range(75, 90, 5)],
    'colsample_bytree': [i/10.0 for i in range(75, 90, 5)]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_tree6,
            param_grid = param_grid_tree6,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)


# -----------------------------------
# 调节正则化参数
# reg_alpha, reg_lambda
# -----------------------------------
param_xgb_regu1 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_regu1 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_regu1,
            param_grid = param_grid_regu1,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)

param_xgb_regu2 = {
    'learning_rate': 0.1,
    'n_estimators': 140,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': "binary:logistic",
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
param_grid_regu2 = {
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}
scoring = 'roc_auc'
n_jobs = 4
cv_method = 5

grid_search(train = train,
            predictors = predictors,
            target = target,
            param_xgb = param_xgb_regu2,
            param_grid = param_grid_regu2,
            scoring = scoring,
            n_jobs = n_jobs,
            cv_method = cv_method)



# -----------------------------------
param_xgb3 = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.005,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model3 = XGBClassifier(**param_xgb3)

modelFit(alg = xgb_model3,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)



# -----------------------------------
# 降低learning rate
# 增加n_estimators
# -----------------------------------
param_xgb4 = {
    'learning_rate': 0.1,
    'n_estimators': 5000,
    'max_depth': 4,
    'min_child_weight': 6,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.005,
    'objective': 'binary:logistic',
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 27
}
scoring = "auc"
cv_method = 5
early_stopping_rounds = 50
xgb_model4 = XGBClassifier(**param_xgb4)

modelFit(alg = xgb_model4,
        dtrain = train,
        predictors = predictors,
        target = target,
        scoring = scoring,
        useTrainCV = True,
        cv_folds = cv_method,
        early_stopping_rounds = early_stopping_rounds)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
