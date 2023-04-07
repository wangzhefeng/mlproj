# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from get_data import get_lgb_train_test_data
from get_data import data_path
from get_data import model_saved_path
from custom_metrics import binary_error, accuracy
from custom_object_function import loglikelihood


try:
    import cPickle as pickle
except BaseException:
    import pickle

"""
#TODO 1.特征工程
#TODO 2.模型参数选择(交叉验证、GridSearch)
#TODO 3.过拟合处理
"""

# ------------------------------
# data
# ------------------------------
print("Loading data...")
train_path = os.path.join(data_path, "lgb_data/binary_classification/binary.train")
test_path = os.path.join(data_path, "lgb_data/binary_classification/binary.test")
weight_path = [
    os.path.join(data_path, "lgb_data/binary_classification/binary.train.weight"), 
    os.path.join(data_path, "lgb_data/binary_classification/binary.train.weight")
]
W_train, W_test, X_train, y_train, X_test, y_test, lgb_train, lgb_eval = get_lgb_train_test_data(
    train_path, test_path, weight_path
)
num_train, num_feature = X_train.shape
feature_name = ["feature_" + str(col) for col in range(num_feature)]
print(f"W_train.head():\n {W_train.head()}")
print()
print(f"W.train.shape:\n {W_train.shape}")
print()
print(f"X_train.head():\n {X_train.head()}")
print()
print(f"X_train.shape:\n {X_train.shape}")
print()
print(f"num_train:\n {num_train}")
print()
print(f"num_feature:\n {num_feature}")
print()
print(f"feature_name:\n {feature_name}")
print()

# ------------------------------
# model parameters
# ------------------------------
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

# ------------------------------
# model training
# ------------------------------
print("Starting training...")
gbm = lgb.train(
    params, 
    lgb_train, 
    num_boost_round = 10,
    valid_sets = lgb_train, 
    feature_name = feature_name,
    categorical_feature = [21]
)
print("Finished first 10 round...")
print("7th feature name is:", lgb_train.feature_name[6])

# ------------------------------
# model save
# ------------------------------
print("Saving ")
#TODO
gbm.save_model("model.txt")


# ------------------------------
#TODO
# ------------------------------
print("Dumping model to JSON...")
model_json = gbm.dump_model()
#TODO
with open("model.json", "w+") as f:
    json.dump(model_json, f, indent = 4)

print(f"Feature names: {gbm.feature_name()}")
print(f"Feature importances: {list(gbm.feature_importance)}")

# ------------------------------
# model predict
# ------------------------------
print("Loading model to predict...")
#TODO
bst = lgb.Booster(model_file = model.txt)
y_pred = bst.predict(X_test)
print(f"The rmse of loaded model's prediction is: {mean_squared_error(y_test, y_pred) ** 0.5}")



# ------------------------------
#
# ------------------------------
print("Dumpong and loading model with pickle...")
#TODO
with open("model.pkl", "wb") as fout:
    pickle.dump(gbm, fout)

#TODO
with open("model.pkl", "rb") as fin:
    pkl_bst = pickle.load(fin)

y_pred = pkl_bst.predict(X_test, num_iteration = 7)
print(f"The rmse of pickled model's prediction is: {mean_squared_error(y_test, y_pred) ** 0.5}")








gbm = lgb.train(params, 
                lgb_train,
                num_boost_round = 10,
                init_model = "model.txt",
                valid_sets = lgb_eval)
print("Finished 10 - 20 round with model file...")


gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 10,
                init_model = gbm,
                learning_rates = lambda iter: 0.05 * (0.99 ** iter),
                valid_sets = lgb_eval)
print('Finished 20 - 30 rounds with decay learning rates...')


gbm = lgb.train(params, 
                lgb_train,
                num_boost_round = 10,
                init_model = gbm,
                valid_sets = lgb_eval,
                callbacks = [lgb.reset_parameter(bagging_fraction = [0.7] * 5 + [0.6] * 5)])
print('Finished 30 - 40 rounds with changing bagging_fraction...')


gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 10,
                init_model = gbm,
                fobj = loglikelihood,
                feval = binary_error,
                valid_sets = lgb_eval)
print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')

gbm = lgb.train(params,
                lgb_trian,
                num_boost_round = 10,
                init_model = gbm,
                fobj = loglikelihood,
                feval = [binary_error, accuracy],
                valid_sets = lgb_eval)
print('Finished 50 - 60 rounds with self-defined objective function and multiple self-defined eval metrics...')





print('Starting a new training job...')
# callback
def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference = lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print("Add a new valid dataset at iteration 5...")
            env.model.add_valid(lgb_eval_new, "new_valid")
    callback.before_iteration = True
    callbakc.order = 0

    return callback

gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 10,
                valid_sets = lgb_train,
                callbacks = [reset_metrics()])
print("Finished first 10 rounds with callback function...")