# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
# 标准化
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import scale
# 特征缩放到一个范围
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
# 归一化
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
# 特征稳健缩放(存在异常值特征)
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale
# 分箱离散化
from sklearn.preprocessing import Binarizer
# from sklearn.preprocessing import binarize
# from sklearn.preprocessing import KBinDiscretizer
# 多项式转换
from sklearn.preprocessing import PolynomialFeatures
# 对数转换
from sklearn.preprocessing import FunctionTransformer
# 类别型特征重新编码
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import normalize
# from sklearn.preprocessing import Quantiletransformer
# from sklearn.preprocessing import Powertransformer
from sklearn.preprocessing import label_binarize
# from sklearn.compose import transformedTargetRegressor
from scipy.stats import skew


# ================================================
# 数值型变量分布转换
# ================================================
def standard_center(features, is_copy = True, with_mean = True, with_std = True):
    """
    DONE
    标准化/方差缩放
    """
    ss = StandardScaler(copy = is_copy, with_mean = with_mean, with_std = with_std)
    transformed_data = ss.fit_transform(features)
    return transformed_data


def normalizer_min_max(features):
    """
    DONE
    归一化--区间缩放
        min-max 区间缩放
    """
    mms = MinMaxScaler()
    transformed_data = mms.fit_transform(features)
    return transformed_data


def normalizer_min_max_feature(feature):
    """
    DONE
    归一化--区间缩放
        min-max 区间缩放
            Box-Cox 变换之前
    """
    transformed_data = (feature - feature.min()) / (feature.max() - feature.min())
    return transformed_data


def normalizer_L2(features):
    """
    DONE
    归一化--将样本的特征值转化到同一量刚下
        把数据映射到 [0,1]或者[a,b]
            L2
    """
    norm = Normalizer()
    transformed_data = norm.fit_transform(features)
    return transformed_data


def normalizer_Ln(features, norm, axis, is_copy = True, return_norm = False):
    """
    DONE
    正则化: 将每个样本或特征正则化为L1, L2范数
    """
    transformed_data = normalize(
        X = features,
        norm = norm,
        axis = axis,
        copy = is_copy,
        return_norm = return_norm
    )
    return transformed_data


def robust_tansform(features):
    """
    稳健缩放
    """
    rs = RobustScaler()
    transformed_data = RobustScaler(features)
    return transformed_data


def log_transform_feature(feature):
    """
    对数转换
    """
    transformed_data = np.log1p(feature)
    return transformed_data


def log1p_transform(features):
    """
    对数转换
    """
    ft = FunctionTransformer(np.log1p, validate = False)
    transformed_data = ft.fit_transform()
    return transformed_data


def box_cox_transform(features):
    """
    Box-Cox 转换
    """
    bc = Powertransformer(method = "box-cox", standardize = False)
    transformed_data = bc.fit_transform(features)
    return transformed_data


def yeo_johnson_transform(features):
    """
    Yeo-Johnson 转换
    """
    yj = Powertransformer(method = "yeo-johnson", standardize = False)
    transformed_data = yj.fit_transform(features)
    return transformed_data


def ploynomial_transform(features):
    """
    多项式转换
    """
    pn = PolynomialFeatures()
    transformed_data = pn.fit_transform(features)
    return transformed_data
# ================================================
# 类别性特征重编码
# ================================================
def oneHotEncoding(data, limit_value = 10):
    """
    One-Hot Encoding: pandas get_dummies
    """
    feature_cnt = data.shape[1]
    class_index = []
    class_df = pd.DataFrame()
    normal_index = []
    for i in range(feature_cnt):
        if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
            class_index.append(i)
            class_df = pd.concat([class_df, pd.get_dummies(data.iloc[:, i], prefix = data.columns[i])], axis = 1)
        else:
            normal_index.append(i)
    data_update = pd.concat([data.iloc[:, normal_index], class_df], axis = 1)
    return data_update


def order_encoder(feature):
    enc = OrdinalEncoder()
    encoded_feats = enc.fit_transform(feature)

    return encoded_feats


def one_hot_encoder(feature):
    """
    One-Hot Encoding: sklearn.preprocessing.OneHotEncoder
    """
    enc = OneHotEncoder(categories = "auto")
    encoded_feature = enc.fit_transform(feature)
    return encoded_feature
# ================================================
# 数值型变量分箱离散化
# ================================================
def binarization(features, threshold = 0.0, is_copy = True):
    """
    DONE
    数值特征二值化
    """
    bined = Binarizer(threshold = threshold, copy = is_copy)
    transformed_data = bined.fit_transform(features)
    return transformed_data


def k_bins(data, n_bins, encoder = "ordinal", strategy = "quantile"):
    """
    分箱离散化
    * encode:
        - "ordinal"
        - "onehot"
        - "onehot-dense"
    * strategy:
        - "uniform"
        - "quantile"
        - "kmeans"
    """
    est = preprocessing.KBinsDiscretizer(n_bins = n_bins, encoder = encoder, strategy = strategy)
    transformed_data = est.fit_transform(data)

    return transformed_data
# ================================================
# 数值型变量分箱离散化
# ================================================
def feature_hist(feature):
    mpl.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({
        '%s' % feature: feature,
        'log(1 + %s)' % feature: log_trans_norm(feature)
    })
    prices.hist()


def normality_transform(feature):
    """
    # Map data from any distribution to as close to Gaussian distribution as possible
    # in order to stabilize variance and minimize skewness:
    #   - log(1 + x) transform
    #   - Yeo-Johnson transform
    #   - Box-Cox transform
    #   - Quantile transform
    """
    pass


def quantileNorm(feature):
    qt = Quantiletransformer(output_distribution = "normal", random_state = 0)
    feat_trans = qt.fit_transform(feature)

    return feat_trans


def quantileUniform(feature, feat_test = None):
    qu = Quantiletransformer(random_state = 0)
    feat_trans = qu.fit_transform(feature)
    feat_trans_test = qu.transform(feat_test)

    return feature, feat_trans_test


def feature_dtype(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.dtypes)


def numeric_categorical_features(data, limit_value = 0):
    columns = data.columns

    num_feature_idx = []
    cate_feature_idx = []
    for i in columns:
        if (data[i].dtypes != "object") & (len(set(data[i])) >= limit_value):
            num_feature_idx.append(i)
        else:
            cate_feature_idx.append(i)

    num_feat_index = data[num_feature_idx].columns
    num_feat = data[num_feature_idx]
    cate_feat_index = data[cate_feature_idx].columns
    cate_feat = data[cate_feature_idx]

    return num_feat, num_feat_index, cate_feat, cate_feat_index


def skewed_features(data, num_feat_idx, limit_value = 0.75):
    skewed_feat = data[num_feat_idx].apply(lambda x: skew(x.dropna()))
    skewed_feat = skewed_feat[np.abs(skewed_feat) > limit_value]
    skewed_feat_index = skewed_feat.index

    return skewed_feat, skewed_feat_index


def targettransformer():
    trans = Quantiletransformer(output_distribution = "normal")
    return trans


def binarize_label(y, classes_list):
    y = label_binarize(y, classes = classes_list)

    return y





if __name__ == "__main__":
    pass
