# -*- coding: utf-8 -*-

__author__ = 'wangzhefeng'


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA


# datasets
X, y = make_blobs(n_samples = 10000, 
				  n_features = 3, 
				  centers = [[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]], 
				  cluster_std = [0.2, 0.1, 0.2, 0.2],
				  random_state = 9)
print(X)
print(y)
# fig = plt.figure()
# ax = Axes3D(fig, rect = [0, 0, 1, 1], elev = 30, azim = 20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker = 'o')
# plt.show()


# n_components = 3
pca = PCA(n_components = 3)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)


# n_components = 2
pca = PCA(n_components = 2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

X_new = pca.transform(X)
# plt.scatter(X_new[:, 0], X[:, 1], marker = 'o')
# plt.show()


# n_components = 0.95
pca_percent95 = PCA(n_components = 0.95)
pca_percent95.fit(X)
print(r"-----------------------")
print(pca_percent95.explained_variance_ratio_)
print(pca_percent95.explained_variance_)
print(pca_percent95.n_components_)


# n_components = 0.99
pca_percent99 = PCA(n_components = 0.99)
pca_percent99.fit(X)
print(r"-----------------------")
print(pca_percent99.explained_variance_ratio_)
print(pca_percent99.explained_variance_)
print(pca_percent99.n_components_)


# n_components = 'mle'
# pca_mle = PCA(n_components = 'mle')
# pca_mle.fit(X)
# print(r"------------------------")
# print(pca_mle.explained_variance_ratio_)
# print(pca_mle.explained_variance_)
# print(pca_mle.n_components_)

