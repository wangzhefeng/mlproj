# -*- coding: utf-8 -*-
from sklearn.preprocessing import PolynomialFeatures


"""
特征工程常用方法：
    - 构造多项式特征
    - 组合现有特征
"""


class FeatureBuilding:
    """
    特征生成
    """
    def __init__(self, data):
        self.data = data


    def polynomial(self, degree = 3, is_interaction_only = True, is_include_bias = True):
        """
        生成多项式特征
        Args:
            degree:
            is_interaction_only:
            is_include_bias:
        """
        pf = PolynomialFeatures(degree = degree, interaction_only = is_interaction_only, include_bias = is_include_bias)
        self.transformed_data = pf.fit_transform(self.data)
