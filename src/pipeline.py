# -*- coding: utf-8 -*-


# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# **********************************************


# python libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import linearSVC, SVC
# 缺失值填充
from sklearn.impute import SimpleImputer, MissingIndicator

# scale_center
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Scaling features to a range,like [0, 1]
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

# Scaling features to a range [-1, 1]
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import maxabs_scale

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# global variable
GLOBAL_VARIABLE = None


# =============================================================
# data
# =============================================================
data_url = ("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
data = pd.read_csv(data_url)

# train and test dataset
X_train = []
y_train = []
X_test = []
y_test = []
# =============================================================
# 数值特征转换与处理[scalers, transformers, normalizers]
# =============================================================
"""
* 需要标准化(Standardization: Gaussian with zero mean and unit varianve)的模型: 
    - 带正则化的模型
        - Ridge
        - Lasso
        - ElasticNet
        - Logistic with L1 or L2
        - SVM with RBF kernel and L1 or L2
    - PCA
- 需要MinMaxScaler的特征
    - 标准差较小的特征
    - 稀疏特征(很多元素是0)
"""
# number features
numeric_features = []
numeric_transformer = [
    ("missing_imputer_mean", SimpleImputer(missing_values = np.nan, strategy = "mean")),
    ("missing_imputer_median", SimpleImputer(missing_values = np.nan, strategy = "median")),
    ("missing_imputer_mode", SimpleImputer(missing_values = np.nan, strategy = "most_frequent")),
    ("missing_imputer_constant", SimpleImputer(missing_values = -1, strategy = "constant", fill_value = 0)),

    ("stanadard_scaler", StandardScaler()),
    ("min_max_scaler", MinMaxScaler(feature_range = (0, 1))),
    ("max_abs_scaler", MaxAbsScaler()),
    ("robust_scaler", RobustScaler(quantile_range = (25, 75))),
    ("power_scaler", PowerTransformer(method = "yeo-johnson")),
    ("power_scaler", PowerTransformer(method = "box-cox")),
    ("quantile_scaler", QuantileTransformer(output_distribution = "normal")),
    ("uniform_quantile_scaler", QuantileTransformer(output_distributions = "uniform")),
    ("normalizer_scaler", Normalizer()),

    ("binning", KBinsDiscretizer(n_bins = [], encode = "ordinal")),
    ("binarizer", Binarizer())
]
# =============================================================
# 类别特征转换与处理
# =============================================================
cate_features = []
cate_transformer = [
    ("missing_imputer_mean", SimpleImputer(missing_values = np.nan, strategy = "mean")),
    ("missing_imputer_median", SimpleImputer(missing_values = np.nan, strategy = "median")),
    ("missing_imputer_mode", SimpleImputer(missing_values = np.nan, strategy = "most_frequent")),
    ("missing_imputer_constant", SimpleImputer(missing_values = -1, strategy = "constant", fill_value = 0)),
]
# =============================================================
# 降维
# =============================================================

# =============================================================
# 特征工程
# =============================================================

# =============================================================
# 模型超参数调整
# =============================================================
clf = None
param_grid = {
    "perprocessor_num_imputer_strategy": ["mean", "median"],
    "classifier_c": [0.1, 1.0, 10, 100],
}
param_gird = dict(
    reduce_dim__n_components = [2, 5, 10],
    clf__C = [0.1, 10, 100]
)
pipeline = None

grid_search_1 = GridSearchCV(clf, param_grid, cv = 10, iid = False)
grid_search_2 = GridSearchCV(pipeline, param_grid = param_grid)
grid_search_1.fit(X_train, y_train)
# =============================================================
# Pipeline
# =============================================================
estimators = [
    ("PCA_reduce_dim", PCA()),
    ("SVC_clf", SVC())
]

pipeline = Pipeline(steps = estimators)
pipeline_man = make_pipeline(
    Binarizer(),
)

print(pipeline.setps[0])
print(pipeline.named_steps["PCA_reduce_dim"])
print(pipeline.set_params(SVC_clf__C = 10))




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()



