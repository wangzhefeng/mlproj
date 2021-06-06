# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, NuSVC, LinearSVC


# =======================================================
# data
# =======================================================
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
													test_size = 0.4,
													random_state = 27149)



# cross_val_score
svc = SVC(kernel = "linear", C = 1)
scores = cross_val_score(svc, X_train, y_train, cv = 5)
scores_f1_macro = cross_val_score(svc, X_train, y_train, cv = 5, scoring = "f1_macro")

print("模型在5折交叉验证的准确率：%s" % scores)
print("模型通过5折交叉验证的平均准确率为: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("模型在5折交叉验证的f1-macro：%s" % scores)
print("模型通过5折交叉验证的平均f1-macro为: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




def svc_model(kernel = 'rbf', C = 1.0, nu = 0.5,
			  poly_degree = 3, gamma = 'auto', coef0 = 0,
			  shrinking = True, probability = False, tol = 1e-3,
			  cache_size = None,
			  class_weight = None,
			  verbose = False,
			  max_iter = -1,
			  decision_function_shape = 'ovr',
			  random_state = None):
	""""""
	if kernel == 'poly':
		svc = SVC(C = C, kernel = kernel, degree = poly_degree, gamma = gamma, coef0 = coef0,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, degree = poly_degree, gamma = gamma, coef0 = coef0,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'rbf':
		svc = SVC(C = C, kernel = kernel, gamma = gamma,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, gamma = gamma,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'sigmoid':
		svc = SVC(C = C, kernel = kernel, gamma = gamma, coef0 = coef0,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, gamma = gamma, coef0 = coef0,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'precomputed':
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'linear':
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
		linear_svc = LinearSVC(penalty = 'l2', loss = 'square_hinge',
							   dual = True, tol = 0.0001, C = 1.0,
							   multi_class = 'ovr',
							   fit_intercept = True, intercept_scaling = 1,
							   class_weight = None,
							   verbose = 0,
							   random_state = None,
							   max_iter = 1000)
	else:
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)




# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC


# ========================================================================================================
# data
# ========================================================================================================
# ----------------------------
# full data
# ----------------------------
iris = datasets.load_iris()
X, y = iris.data, iris.target
y = (y == 0).astype("int32")


# ----------------------------
# create testing dataset
# ----------------------------
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)



# ========================================================================================================
# data preprocessing
# 1.data StandardScaler
# ========================================================================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# ========================================================================================================
# basic models
# ========================================================================================================
svm_l1 = SVC()
svm_l2 = SVC()






# ========================================================================================================
# Parameter tuning
# ========================================================================================================
# ===================================================
# Number of random trials
# ===================================================
NUM_TRIALS = 30



# ===================================================
# cv methods
# ===================================================
cv_5_folds = 5
cv_10_folds = 10

cv_shufflesplit = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 29)

def custom_cv_2folds(X):
	n = X.shape[0]
	i = 1
	while i <= 2:
		idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype = int)
		yield idx, idx
		i += 1
cv_custom = custom_cv_2folds(X)



# ===================================================
# scoring metric
# ===================================================
accuracy = "accuracy"
balanced_accuracy = "balanced_accuracy"
average_precision = "average_precision"
brier_score_loss = "brier_score_loss"
precision = "precision"
precision_micro = "precision_micro"
precision_macro = "precision_macro"
recall = "recall"
recall_micro = "recall_micro"
recall_macro = "recall_macro"
f1 = "f1"
f1_micro = "f1_micro"
f1_macro = "f1_macro"
f1_weighted = "f1_weighted"
f1_samples = "f1_samples"
neg_log_loss = "neg_log_loss"
roc_auc = "roc_auc"

scoring = ["accuracy",
		   "precision", "precision_micro", "precision_macro",
		   "recall", "recall_micro", "recall_macro",
		   "f1", "f1_micro", "f1_macro",
		   "roc_auc"]

scoring_custom = {
	"accuracy": "accuracy",
	"precision": "precision",
	"precision_micro": "precision_micro",
	"precision_macro": make_scorer(precision_score, average = "macro"),
	"recall": "recall",
	"recall_micro": "recall_micro",
	"recall_macro": make_scorer(recall_score, average = "macro"),
	"f1": "f1",
	"f1_micro": "f1_mircro",
	"f1_macro": make_scorer(f1_macro, average = "macro"),
	"roc_auc": "roc_auc"
}

# ===================================================
# parameters
# ===================================================
param_grid_svc_l1 = [
	{"C": [1, 10, 100, 1000], "gamma": [0.0001, 0.001, 0.01, 0.1], "kernel": ["rbf"]},
	{"C": [1, 10, 100, 1000], "kernel": ["linear"]},
	{"C": [1, 10, 100, 1000], "gamma": [0.0001, 0.001, 0.01, 0.1],
	 "degree": [3, 4, 5, 6, 7, 8], "kernel": ["poly"]},
	{"C": [], "kernel": ["sigmoid"]},
]



for i in range(NUM_TRIALS):
	cv_inner = KFold(n_splits = cv_5_folds, shuffle = True, random_state = 29)
	cv_outer = KFold(n_splits = cv_5_folds, shuffle = True, random_state = 29)

	clf = GridSearchCV(estimator = svm_l1,
					   param_grid = param_grid_svc_l1,
					   scoring = scoring,
					   n_jobs = None,
					   refit = True,
					   cv = cv_inner,
					   return_train_score = True)
	clf.fit(X, y)
	non_nested_scores[i] = clf.best_score_
	cv_result[i] = clf.cv_results_
	best_estimator[i] = clf.best_estimator_
	best_index[i] = clf.best_index_
	best_params[i] = clf.best_params_
	best_score[i] = clf.best_score_
	scorer[i] = clf.scorer_
	n_splits[i] = clf.n_splits_
	refit_time[i] = clf.refit_time_
	# nested_score = cross_validate(clf, X, y,
	# 									   cv = cv,
	# 									   scoring = scoring,
	# 									   return_train_score = True)
	# nested_score[i] = nested_score.mean()






