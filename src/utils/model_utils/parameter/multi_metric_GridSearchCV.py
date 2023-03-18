# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X, y = make_hastie_10_2(n_samples = 8000, random_state = 42)

param_grid = [{
 "min_samples_split": range(2, 403, 10)
}]

scoring = {
	"AUC": "roc_auc",
	"Accuracy": make_scorer(accuracy_score)
}



dtc = DecisionTreeClassifier(random_state = 42)

gs = GridSearchCV(estimator = dtc,
				  param_grid = param_grid,
				  scoring = scoring,
				  cv = 5,
				  refit = "AUC",
				  return_train_score = True)
gs.fit(X, y)
results = gs.cv_results_
print(results)