# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn import datasets



dataset = datasets.fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
X = X_full[:, [0, 5]]

distributions = [
	('Unscaled data', X),
	('Data after standard scaling', StandardScaler().fit_transform(X)),
	('Data after min-max scaling', MinMaxScaler().fit_transform(X)),
	('Data after max-abs scaling', MaxAbsScaler().fit_transform(X)),
	('Data after robust scaling', RobustScaler(quantile_range = (25, 75)).fit_transform(X)),
	('Data after power transformation (Yeo-Johnson)', PowerTransformer(method = 'yeo-johnson').fit_transform(X)),
	('Data after power transformation (Box-Cox)', PowerTransformer(method = 'box-cox').fit_transform(X)),
	('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution = 'normal').fit_transform(X)),
	('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution = 'uniform').fit_transform(X)),
	('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X)),
]

y = minmax_scale(y_full)

