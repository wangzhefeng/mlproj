# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


from sklearn.metrics import make_scorer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix



class cluster_score:
	def __init__(self, labels_true, labels_pred, X = None, metric = "euclidean"):
		self.labels_true = labels_true
		self.labels_pred = labels_pred
		self.X = X
		self.metric = metric

	def adjusted_rand_index(self):
		score = adjusted_rand_score(self.labels_true, self.labels_pred)

		return score

	def mutual_info(self):
		score = mutual_info_score(self.labels_true, self.labels_pred)

		return score

	def adjust_mutual_info(self):
		score = adjusted_mutual_info_score(self.labels_true, self.labels_pred)

		return score

	def normalized_mutual_info(self):
		score = normalized_mutual_info_score(self.labels_true, self.labels_pred)

		return score

	def homogeneity(self):
		score = homogeneity_score(self.labels_true, self.labels_pred)

		return score

	def completeness(self):
		score = completeness_score(self.labels_true, self.labels_pred)

		return score

	def v_measure(self):
		score = v_measure_score(self.labels_true, self.labels_pred)

		return score

	def homogeneity_completeness_v_measure(self):
		homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(self.labels_true, self.labels_pred)

		return homogeneity, completeness, v_measure

	def Fowlkes_Mallows(self):
		score = fowlkes_mallows_score(self.labels_true, self.labels_pred)

		return score

	def Silhouette_Coefficient(self):
		score = silhouette_score(self.X, self.labels_pred, self.metrics)

		return score

	def Calinski_Harabaz_index(self):
		score = calinski_harabaz_score(self.X, self.labels_pred)

		return score

	def Davies_Bouldin_index(self):
		score = davies_bouldin_score(self.X, self.labels_pred)

		return score

	def Contingency_Matrix(self):
		score = contingency_matrix(self.labels_true, self.labels_pred)

		return score

