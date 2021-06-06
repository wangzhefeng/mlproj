# -*- coding: utf-8 -*-



import numpy as np
from sklearn import datasets
from sklearn.cluster import FeatureAgglomeration


"""
class sklearn.cluster.FeatureAgglomeration(n_clusters = 2,
										   affinity = "euclidean",
										   memory = None,
										   connectivity = None,
										   compute_full_tree = "auto",
										   linkage = "ward",
										   pooling_func = <function mean>)
"""

# ========================================================================
# data
# ========================================================================
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
print(X.shape)


# ========================================================================
# 降维
# ========================================================================
agglo = FeatureAgglomeration(n_clusters = 32)
agglo.fit(X)
print(agglo.labels_)
print(agglo.n_leaves_)
print(agglo.n_components_)
print(agglo.children_)

X_reduced = agglo.transform(X)
print(X_reduced.shape)
