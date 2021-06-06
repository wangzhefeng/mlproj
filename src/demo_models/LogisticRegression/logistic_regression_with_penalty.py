# -*- coding: utf-8 -*-


from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline


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
# logistic regression with l1 penalty
lr_l1_liblinear = LogisticRegression(penalty = "l1", solver = "liblinear")
lr_l1_saga = LogisticRegression(penalty = "l1", solver = "saga")

# logistic regression with l2 penalty
lr_l2_newton = LogisticRegression(penalty = "l2", solver = "newton-cg")
lr_l2_lbfgs = LogisticRegression(penalty = "l2", solver = "lbfgs")
lr_l2_liblinear = LogisticRegression(penalty = "l2", solver = "liblinear")
lr_l2_sag = LogisticRegression(penalty = "l2", solver = "sag")
lr_l2_saga = LogisticRegression(penalty = "l2", solver = "saga")

# logistic regression using sgd algorithm
lr_sgd_l1 = SGDClassifier(loss = "log", penalty = "l1")
lr_sgd_l2 = SGDClassifier(loss = "log", penalty = "l2")
lr_sgd_elsasticnet = SGDClassifier(loss = "log", penalty = "elasticnet")


# ========================================================================================================
# basic model
# hold-out
# ========================================================================================================
# split dataset to train and validate dataset
X_train, X_val, y_train, y_val = train_test_split(X, y)

# train the lr with train dataset
lr_l1_liblinear.fit(X_train, y_train)

# prediction and scores on validate dataset
y_pred_class = lr_l1_liblinear.predict(X_val)
y_pred_proba = lr_l1_liblinear.predict_proba(X_val)
y_pred_log_proba = lr_l1_liblinear.predict_log_proba(X_val)
pprint(y_pred_class)
pprint(y_pred_proba)
pprint(y_pred_log_proba)

# scores on train dataset
score_train = lr_l1_liblinear.score(X_train, y_train)
pprint(score_train)

# scores on validation dataset
validate_score = lr_l1_liblinear.score(X_val, y_val)
pprint(validate_score)


# ========================================================================================================
# cross_val_score, cross_validate, cross_val_predict
# ========================================================================================================
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
# train model with `cross_val_scores`
# ===================================================
cv_accuracy_scores_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = accuracy)
# clf = make_pipeline(
# 	lr_l1_liblinear,
# )
# cv_accuracy_scores_make_pipeline = cross_val_score(clf, X, y, cv = cv_10_folds, scoring = accuracy)
cv_balanced_accuracy_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = balanced_accuracy)
cv_average_precision_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = average_precision)
cv_brier_score_loss_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = brier_score_loss)
cv_precision_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = precision)
cv_recall_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = recall)
cv_f1_scores_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = f1)
cv_f1_micro_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = f1_micro)
cv_f1_macor_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = f1_macro)
cv_f1_weighted_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = f1_weighted)
# cv_f1_samples_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = f1_samples)
cv_neg_log_loss_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = neg_log_loss)
cv_roc_auc_1 = cross_val_score(lr_l1_liblinear, X, y, cv = cv_10_folds, scoring = roc_auc)
pprint("=" * 100)
pprint(cv_accuracy_scores_1)
# pprint(cv_accuracy_scores_make_pipeline)
pprint(cv_balanced_accuracy_1)
pprint(cv_average_precision_1)
pprint(cv_brier_score_loss_1)
pprint(cv_precision_1)
pprint(cv_recall_1)
pprint(cv_f1_scores_1)
pprint(cv_f1_micro_1)
pprint(cv_f1_macor_1)
pprint(cv_f1_weighted_1)
# pprint(cv_f1_samples_1)
pprint(cv_neg_log_loss_1)
pprint(cv_roc_auc_1)



# ===================================================
# train model with "cross_validate"
# ===================================================
cv_scores = cross_validate(lr_l1_liblinear, X, y,
						   cv = cv_10_folds,
						   scoring = scoring,
						   return_train_score = True,
						   return_estimator = True)
pprint("=" * 100)
pprint(sorted(cv_scores.keys()))
pprint(cv_scores["estimator"])
pprint(cv_scores["fit_time"])
pprint(cv_scores["score_time"])
pprint(cv_scores["test_accuracy"])
pprint(cv_scores["train_accuracy"])



# ======================================================================================================
# cross validation iterators
# ======================================================================================================
# ===================================================
# KFold
# RepeatedKFold
# ===================================================
cv = KFold(n_splits = cv_5_folds)
for i, (train, validate) in enumerate(cv.split(X = X, y = y)):
	print("分裂次数: ", i)
	print("Train Index: ", train)
	print("Validate Index: ", validate)
	indices = np.array([np.nan] * len(X))
	indices[train] = 1
	indices[validate] = 0
	print("KFold splited Index: ", indices)

repeated_kfold_cv = RepeatedKFold(n_splits = cv_5_folds, n_repeats = 100)




# ===================================================
# StratifiedKFold
# RepeatedStratifiedKFold
# StratifiedShuffleSplit
# ===================================================










# ========================================================================================================
# GridSearchCV, RandomizedSearchCV
# ========================================================================================================
# Tips for parameter search
	# specifying an objective metric
	# specifying multiple metircs for evaluation
	# composite estimators and parameter spaces
	# Model selection: development and evalution
	# Parallelism
	# Robustness to failure

param_grid_lr_l1 = {
	"penalty": ["l1"],
	"dual": [False],
	"tol": [10e-4, 10e-5, 10e-6],
	"C": [0.001, 0.01, 0.1, 1.0, 10],
	"solver": ["liblinear", "saga"],
}

param_grid_lr_l2 = {
	"penalty": ["l2"],
	"dual": [False],
	"tol": [10e-4, 10e-5, 10e-6],
	"C": [0.001, 0.01, 0.1, 1.0, 10],
	"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

param_random = {

}

def GridSearch_CV():
	gscv = GridSearchCV(estimator = model,
						param_grid = param_grid,
						scoring = scoring,
						n_jobs = None,
						refit = is_refit,
						cv = cv_inner,
						return_train_score = True)
	gscv.fit(X, y)
	# cv_result = gscv.cv_results_
	# best_estimator = gscv.best_estimator_
	# best_index = gscv.best_index_
	# best_params = gscv.best_params_
	# best_score = gscv.best_score_
	# scorer = gscv.scorer_
	# n_splits = gscv.n_splits_
	# refit_time = gscv.refit_time_
	nested_scores = cross_validation(gscv, X, y,
									 method = method_cv,
									 cv = cv_outer,
									 scoring = scoring)
	gen_scores = nested_scores.mean()














# ==============================================================================
# logistic regression with cross validation
lr_cv_l1_liblinear = LogisticRegressionCV(penalty = "l1", solver = "liblinear")
lr_cv_l1_saga = LogisticRegressionCV(penalty = "l1", solver = "saga")
lr_cv_l2_newton = LogisticRegressionCV(penalty = "l2", solver = "newton-cg")
lr_cv_l2_lbfgs = LogisticRegressionCV(penalty = "l2", solver = "lbfgs")
lr_cv_l2_liblinear = LogisticRegressionCV(penalty = "l2", solver = "liblinear")
lr_cv_l2_sag = LogisticRegressionCV(penalty = "l2", solver = "sag")
lr_cv_l2_saga = LogisticRegressionCV(penalty = "l2", solver = "saga")
