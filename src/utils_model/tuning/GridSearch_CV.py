# -*- coding: utf-8 -*-

"""
@author: tinker
@date: 2019-02-01
"""

from __future__ import print_function
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import classification_report

from sklearn import datasets

from sklearn import svm

from sklearn.pipeline import make_pipeline


print(__doc__)

# =================================================================
# data
# =================================================================
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


tuned_parameters = [
	{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

scores = ["precision", "recall"]

for score in scores:
	print("# ----------------------------------------------")
	print("# Tuning hyper-parameters for %s" % score)

	print("# ----------------------------------------------")
	print("# Best parameters set found on development set:")

	clf = GridSearchCV(estimator = svm.SVC(),
					   param_grid = tuned_parameters,
					   cv = 5,
					   scoring="%s_macro" % score)
	clf.fit(X_train, y_train)
	print(clf.best_params_)

	print("# ----------------------------------------------")
	print("# Grid scores on development set:")
	print("# ----------------------------------------------")
	means = clf.cv_results_["mean_test_score"]
	stds = clf.cv_results_["std_test_score"]
	for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))
	print("# ----------------------------------------------")
	print("# Detailed classification report:")
	print()
	print("# The model is trained on the full development set.")
	print("# The scores are computed on the full evaluation set.")
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))