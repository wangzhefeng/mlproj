# -*- coding: utf-8 -*-
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit


# data
np.random.seed(0)
N = 1000
X = pd.DataFrame({
    "continuous": range(N),
    "categorical": np.repeat([0, 1, 2, 3, 4], N / 5)
})
CATEGORICAL_EFFECTS = [-1, -1, -2, -2, 2]
LINEAR_TEAM = np.array([
    -0.5 + 0.01 * X["continuous"][k] + CATEGORICAL_EFFECTS[X["categorical"][k]] for k in range(X.shape[0])
]) + np.random.normal(0, 1, X.shape[0])
TRUE_PROB = expit(LINEAR_TEAM)
Y = np.random.binomial(1, TRUE_PROB, size = N)
DATA = {
    "X": X,
    "probability_labels": TRUE_PROB,
    "binary_labels": Y,
    "lgb_with_binary_labels": lgb.Dataset(X, Y),
    "lgb_with_probability_labels": lgb.Dataset(X, TRUE_PROB)
}



def experiment(objective, label_type, data):
    """Measure performance of an objective.
    Parameters
    ----------
    objective : string 'binary' or 'xentropy'
        Objective function.
    label_type : string 'binary' or 'probability'
        Type of the label.
    data : dict
        Data for training.
    Returns
    -------
    result : dict
        Experiment summary stats.
    """
    np.random.seed(0)
    nrounds = 5
    lgb_data = data['lgb_with_' + label_type + '_labels']
    params = {
        'objective': objective,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'verbose': -1
    }
    time_zero = time.time()
    gbm = lgb.train(params, lgb_data, num_boost_round=nrounds)
    y_fitted = gbm.predict(data['X'])
    y_true = data[label_type + '_labels']
    duration = time.time() - time_zero
    return {
        'time': duration,
        'correlation': np.corrcoef(y_fitted, y_true)[0, 1],
        'logloss': log_loss(y_fitted, y_true)
    }

#################
# Observe the behavior of `binary` and `xentropy` objectives
print('Performance of `binary` objective with binary labels:')
print(experiment('binary', label_type = 'binary', data = DATA))

print('Performance of `xentropy` objective with binary labels:')
print(experiment('xentropy', label_type = 'binary', data = DATA))

print('Performance of `xentropy` objective with probability labels:')
print(experiment('xentropy', label_type = 'probability', data = DATA))

# Trying this throws an error on non-binary values of y:
#   experiment('binary', label_type='probability', DATA)

# The speed of `binary` is not drastically different than
#   `xentropy`. `xentropy` runs faster than `binary` in many cases, although
#   there are reasons to suspect that `binary` should run faster when the
#   label is an integer instead of a float
K = 10
A = [experiment('binary', label_type = 'binary', data = DATA)['time'] for k in range(K)]
B = [experiment('xentropy', label_type = 'binary', data = DATA)['time'] for k in range(K)]
print('Best `binary` time: ' + str(min(A)))
print('Best `xentropy` time: ' + str(min(B)))