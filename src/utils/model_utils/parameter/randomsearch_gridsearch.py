# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from pprint import pprint

digits = datasets.load_digits()
X, y = digits.data, digits.target

clf = RandomForestClassifier(n_estimators = 20)

param_dist = {
	"max_depth": [3, None],
	"max_features": sp_randint(1, 11),
	"min_samples_split": sp_randint(2, 11),
	"bootstrap": [True, False],
	"criterion": ["gini", "entropy"]
}

param_grid = {
	"max_depth": [3, None],
	"max_features": [1, 3, 10],
	"min_samples_split": [2, 3, 10],
	"bootstrap": [True, False],
	"criterion": ["gini", "entropy"]
}

random_search = RandomizedSearchCV(clf,
								   param_distributions = param_dist,
								   n_iter = 20,
								   cv = 5)
random_search.fit(X, y)



grid_search = GridSearchCV(clf,
						   param_grid = param_grid,
						   cv = 5)
grid_search.fit(X, y)



pprint(random_search.cv_results_)
pprint(random_search.cv_results_.keys())
print()

pprint(grid_search.cv_results_)
pprint(grid_search.cv_results_.keys())