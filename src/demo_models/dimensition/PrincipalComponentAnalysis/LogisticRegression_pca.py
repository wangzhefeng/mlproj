# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



# ==========================================================
# data
# ==========================================================
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target






# ==========================================================
# models
# ==========================================================
lr_clf = SGDClassifier(loss = "log",
					   penalty = "l2",
					   early_stopping = True,
					   max_iter = 10000,
					   tol = "1e-5",
					   random_state = 29)





pca = PCA()

pipe = Pipeline(steps = [
	("pca", pca),
	("logistic", lr_clf)
])

param_grid = {
	"pca__n_components": [5, 20, 30, 40, 50, 64],
	"logistic__alpha": np.logspace(-4, 4, 5)
}

grid_search = GridSearchCV(pipe,
						   param_grid,
						   iid = False,
						   cv = 10,
						   return_train_score = False)




