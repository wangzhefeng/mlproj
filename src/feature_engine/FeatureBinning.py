import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import binarize
from sklearn.preprocessing import KBinDiscretizer
from sklearn.preprocessing import Quantiletransformer
from sklearn.preprocessing import label_binarize
from scipy.stats import skew


"""
数值型变量分箱离散化
"""


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
