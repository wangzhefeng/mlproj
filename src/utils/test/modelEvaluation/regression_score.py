# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

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
from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import make_scorer

from sklearn.metrics import explained_variance_score
# MAE
from sklearn.metrics import mean_absolute_error
# MSE
from sklearn.metrics import mean_squared_error
# MSLE
from sklearn.metrics import mean_squared_log_error
# MAE
from sklearn.metrics import median_absolute_error
# R2
from sklearn.metrics import r2_score



def scoring():
	scoring_regressioner = {
		'R2': r2_score,
		'MES': mean_squared_error,
	}

	return scoring_regressioner

