# -*- coding: utf-8 -*-


from sklearn.decomposition import PCA




def PCA_clf():
	pca = PCA(n_components = None,
			  copy = True,
			  whiten = False,
			  svd_solver = "auto",
			  tol = 0.0,
			  iterated_power = "auto",
			  random_state = None)
	pca.fit(X)
	pca.fit_transform(X)
