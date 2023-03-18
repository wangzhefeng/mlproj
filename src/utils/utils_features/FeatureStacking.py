import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import xgboost
from xgboost import XGBRegressor
import lightgbm
import lightgbm as lgb
from sklearn.model_selection import KFold



def stacking_reg(clf, train_x, train_y, test_x, clf_name, folds, label_split = None):
    """[summary]

    Args:
        clf ([type]): [description]
        train_x ([type]): [description]
        train_y ([type]): [description]
        test_x ([type]): [description]
        clf_name ([type]): [description]
        kf ([type]): [description]
        folds ([type]): [description]
        label_split ([type], optional): [description]. Defaults to None.

    Raises:
        IOError: [description]

    Returns:
        [type]: [description]
    """
    kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pred = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf_reg", "ada_reg", "gb_reg", "et_reg", "lr_reg", "lsvc_reg", "knn_reg"]:
            clf.fit(tr_x, tr_y)
            pred = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pred
            test_pred[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pred))
        elif clf_name in ["xgb_reg"]:
            train_matrix = clf.DMatrix(tr_x, label = tr_y, missing = -1)
            test_matrix = clf.DMatrix(te_x, label = te_y, missing = -1)
            z = clf.DMatrix(test_x, label = te_y, missing = -1)
            params = {
                "booster": "gbtree",
                "eval_metric": "rmse",
                "gamma": 1,
                "min_child_weight": 1.5,
                "max_depth": 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "nthread": 12,
            }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, "train"), (test_matrix, "eval")]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round = num_round, evals = watchlist, early_stopping_rounds = early_stopping_rounds)
                pred = model.predict(test_matrix, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pred
                test_pred[i, :] = model.predict(z, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred))
        elif clf_name in ["lgb_reg"]:
            train_matrix = clf.Dataset(tr_x, label = tr_y)
            test_matrix = clf.Dataset(te_x, label = te_y)
            params = {
                "boosting_type": "gbdt",
                "objective": "regression_l2",
                "metric": "rmse",
                "min_child_weight": 1.5,
                "num_leaves": 2 ** 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "learning_rate": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "nthread": 12,
                "silent": True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round = num_round, valid_sets = test_matrix, early_stopping_rounds = early_stopping_rounds)
                pred = model.predict(te_x, num_iteration = model.best_iteration).reshape(-1, 1)
                train[test_index] = pred
                test_pred[i, :] = model.predict(test_x, num_iteration = model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred))
        else:
            raise IOError("Please add new clf.")
        print(f"{clf_name} now score is:{cv_scores}")
    test[:] = test_pred.mean(axis = 0)
    print(f"{clf_name}_score_list:{cv_scores}")
    cv_scores_mean = np.mean(cv_scores)
    print(f"{clf_name}_score_mean:{cv_scores_mean}")
    return train.reshape(-1, 1), test.reshape(-1, 1)


def stacking_clf(clf, train_x, train_y, test_x, clf_name, folds, label_split = None):
    """[summary]

    Args:
        clf ([type]): [description]
        train_x ([type]): [description]
        train_y ([type]): [description]
        test_x ([type]): [description]
        clf_name ([type]): [description]
        kf ([type]): [description]
        folds ([type]): [description]
        label_split ([type], optional): [description]. Defaults to None.

    Raises:
        IOError: [description]

    Returns:
        [type]: [description]
    """
    kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pred = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf_clf", "ada_clf", "gb_clf", "et_clf", "lr_clf", "knn_clf", "gnb_clf"]:
            clf.fit(tr_x, tr_y)
            pred = clf.predict_proba(te_x)
            train[test_index] = pred[:, 0].reshape(-1, 1)
            test_pred[i, :] = clf.predict_proba(test_x)[:, 0].reshape(-1, 1)
            cv_scores.append(log_loss(te_y, pred[:, 0].reshape(-1, 1)))
        elif clf_name in ["xgb_clf"]:
            train_matrix = clf.DMatrix(tr_x, label = tr_y, missing = -1)
            test_matrix = clf.DMatrix(te_x, label = te_y, missing = -1)
            z = clf.DMatrix(test_x, label = te_y, missing = -1)
            params = {
                "booster": "gbtree",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "gamma": 1,
                "min_child_weight": 1.5,
                "max_depth": 5,
                "lambda": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "eta": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2,
            }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, "train"), (test_matrix, "eval")]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round = num_round, evals = watchlist, early_stopping_rounds = early_stopping_rounds)
                pred = model.predict(test_matrix, ntree_limit = model.best_ntree_limit).reshape(-1, 1)
                train[test_index] = pred[:, 0].reshape(-1, 1)
                test_pred[i, :] = model.predict(z, ntree_limit = model.best_ntree_limit)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss()(te_y, pred[:, 0].reshape(-1, 1)))
        elif clf_name in ["lgb_clf"]:
            train_matrix = clf.Dataset(tr_x, label = tr_y)
            test_matrix = clf.Dataset(te_x, label = te_y)
            params = {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "metric": "multi_logclass",
                "min_child_weight": 1.5,
                "num_leaves": 2 ** 5,
                "lambda_l2": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.7,
                "learning_rate": 0.03,
                "tree_method": "exact",
                "seed": 2017,
                "num_class": 2,
                "silent": True,
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets = test_matrix, early_stopping_rounds = early_stopping_rounds)
                pred = model.predict(te_x, num_iteration = model.best_iteration)
                train[test_index] = pred[:, 0].reshape(-1, 1)
                test_pred[i, :] = model.predict(test_x, num_iteration = model.best_iteration)[:, 0].reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pred[:, 0].reshape(-1, 1)))
        else:
            raise IOError("Please add new clf.")
        print(f"{clf_name} now score is:{cv_scores}")
    test[:] = test_pred.mean(axis = 0)
    print(f"{clf_name}_score_list:{cv_scores}")
    cv_scores_mean = np.mean(cv_scores)
    print(f"{clf_name}_score_mean:{cv_scores_mean}")
    return train.reshape(-1, 1), test.reshape(-1, 1)


"""
regression
"""


def rf_reg(x_train, y_train, x_valid, kf, label_split = None):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs = -1, random_state=2017, max_features="auto", verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid, "rf_reg", kf, label_split=label_split)
    return rf_train, rf_test, "rf_reg"


def ada_reg(x_train, y_train, x_valid, kf, label_split = None):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid, "ada_reg", kf, label_split=label_split)
    return ada_train, ada_test, "ada_reg"


def gb_reg(x_train, y_train, x_valid, kf, label_split = None):
    gbdt = GradientBoostingRegressor(learning_rate = 0.04, n_estimators=100, subsample=0.8, random_state=2017, max_depth = 5, verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid, "gb_reg", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb_reg"


def et_reg(x_train, y_train, x_valid, kf, label_split = None):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=35, max_features="auto", n_jobs = -1, random_state=2017, verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid, "et_reg", kf, label_split=label_split)
    return et_train, et_test, "et_reg"


def lr_reg(x_train, y_train, x_valid, kf, label_split = None):
    lr_reg = LinearRegression(n_jobs = -1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr_reg", kf, label_split=label_split)
    return lr_train, lr_test, "lr_reg"


def xgb_reg(x_train, y_train, x_valid, kf, label_split = None):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid, "xgb_reg", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb_reg"


def lgb_reg(x_train, y_train, x_valid, kf, label_split = None):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid, "lgb_reg", kf, label_split=label_split)
    return lgb_train, lgb_test, "lgb_reg"


"""
classification
"""


def rf_clf(x_train, y_train, x_valid, kf, label_split = None):
    randomforest = RandomForestClassifier(n_estimators = 1200, max_depth = 20, n_jobs = -1, random_state = 2017, max_features = "auto", verbose = 1)
    rf_train, rf_test = stacking_clf(randomforest, x_train, y_train, x_valid, "rf_clf", kf, label_split=label_split)
    return rf_train, rf_test, "rf_clf"


def ada_clf(x_train, y_train, x_valid, kf, label_split = None):
    adaboost = AdaBoostClassifier(n_estimators = 50, random_state = 2017, learning_rate = 0.01)
    ada_train, ada_test = stacking_clf(adaboost, x_train, y_train, x_valid, "ada_clf", kf, label_split=label_split)
    return ada_train, ada_test, "ada_clf"


def gb_clf(x_train, y_train, x_valid, kf, label_split = None):
    gbdt = GradientBoostingClassifier(learning_rate = 0.04, n_estimators = 100, subsample = 0.8, random_state = 2017, max_depth = 5, verbose = 1)
    gbdt_train, gbdt_test = stacking_clf(gbdt, x_train, y_train, x_valid, "gb_clf", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb_clf"


def et_clf(x_train, y_train, x_valid, kf, label_split = None):
    extratree = ExtraTreesClassifier(n_estimators = 1200, max_depth = 35, max_features = "auto", n_jobs = -1, random_state = 2017, verbose = 1)
    et_train, et_test = stacking_clf(extratree, x_train, y_train, x_valid, "et_clf", kf, label_split=label_split)
    return et_train, et_test, "et_clf"


def xgb_clf(x_train, y_train, x_valid, kf, label_split = None):
    xgb_train, xgb_test = stacking_clf(xgboost, x_train, y_train, x_valid, "xgb_clf", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb_clf"


def lgb_clf(x_train, y_train, x_valid, kf, label_split = None):
    lgb_train, lgb_test = stacking_clf(lightgbm, x_train, y_train, x_valid, "lgb_clf", kf, label_split=label_split)
    return lgb_train, lgb_test, "lgb_clf"


def gnb_clf(x_train, y_train, x_valid, kf, label_split = None):
    gnb = GaussianNB()
    gnb_train, gnb_test = stacking_clf(gnb, x_train, y_train, x_valid, "gnb_clf", kf, label_split=label_split)
    return gnb_train, gnb_test, "gnb_clf"


def lr_clf(x_train, y_train, x_valid, kf, label_split = None):
    logisticregression = LogisticRegression(n_jobs = -1, random_state = 2017, C = 0.1, max_iter = 200)
    lr_train, lr_test = stacking_clf(logisticregression, x_train, y_train, x_valid, "lr_clf", kf, label_split=label_split)
    return lr_train, lr_test, "lr_clf"


def knn_clf(x_train, y_train, x_valid, kf, label_split = None):
    kneighbors = KNeighborsClassifier(kneighbors = 200, n_jobs = -1)
    knn_train, knn_test = stacking_clf(kneighbors, x_train, y_train, x_valid, "knn_clf", kf, label_split=label_split)
    return knn_train, knn_test, "knn_clf"
