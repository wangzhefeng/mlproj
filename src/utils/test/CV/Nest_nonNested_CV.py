# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from pprint import pprint
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np



# Number of random trials
NUM_TRIALS = 30

# Load the dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

# We will use a Support Vector Classifier with "rbf" kernel
svm = svm.SVC(kernel = "rbf")




# ===================================================================
# parameter tuning
# ===================================================================
# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

scoring = ["accuracy", "precision", "recall"]
p_grid = {"C": [1, 10, 100], "gamma": [.01, .1]}
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 29)
outer_cv = KFold(n_splits = 5, shuffle = True, random_state = 29)


clf = GridSearchCV(estimator = svm,
				   param_grid = p_grid,
				   cv = inner_cv,
				   scoring = scoring)
clf.fit(X_iris, y_iris)

pprint(clf.cv_results_)
pprint(clf.best_estimator_)
pprint(clf.best_score_)
pprint(clf.best_params_)
pprint(clf.best_index_)
pprint(clf.scorer_)
pprint(clf.n_splits_)
pprint(clf.refit_time_)




# nested_score = cross_val_score(clf,
# 							   X_iris,
# 							   y_iris,
# 							   cv = outer_cv,
# 							   scoring = scoring)
# pprint(nested_score)



