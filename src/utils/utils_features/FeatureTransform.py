# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
# 多项式转换
from sklearn.preprocessing import PolynomialFeatures
# 对数转换
from sklearn.preprocessing import FunctionTransformer
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import normalize
from sklearn.preprocessing import Powertransformer
# from sklearn.compose import transformedTargetRegressor


"""
数值型变量分布转换:
    - 
"""


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

