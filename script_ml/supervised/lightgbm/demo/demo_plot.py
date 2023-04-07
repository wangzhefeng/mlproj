# -*- coding: utf-8 -*-
import os
import lightgbm as lgb
import pandas as pd

from get_data import get_lgb_train_test_data
from get_data import data_path

if lgb.compat.MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
else:
    raise ImportError("You need to install matplotlib for plot_demo.py")

# -------------------------
# data
# -------------------------
print("Loading data....")
train_path = os.path.join(data_path, "lgb_data/regression/regression.train")
test_path = os.path.join(data_path, "lgb_data/regression/regression.test")
X_train, y_train, \
X_test, y_test, \
lgb_train, lgb_test = get_lgb_train_test_data(
    train_path, 
    test_path, 
    weight_paths = []
)

# -------------------------
# model parameters
# -------------------------
params = {
    "num_leaves": 5,
    "metric": ("l1", "l2"),
    "verbose": 0,
}

# -------------------------
# model train
# -------------------------
print("Starting training...")
evals_result = {}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 100,
                valid_sets = [lgb_train, lgb_test],
                feature_name = ["f" + str(i + 1) for i in range(X_train.shape[-1])],
                categorical_feature = [21],
                evals_result = evals_result,
                verbose_eval = 10)

# -------------------------
# 
# -------------------------
# print("Plotting metrics recorded during training...")
# ax = lgb.plot_metric(evals_result, metric = "l1")
# plt.show()

# print("Plotting feature importances...")
# ax = lgb.plot_importance(gbm, max_num_features = 10)
# plt.show()

# print("Plotting split value histogram...")
# ax = lgb.plot_split_value_histogram(gbm, feature = "f26", bins = "auto")
# plt.show()

# print("Plotting 54th tree...")
# ax = lgb.plot_tree(gbm, tree_index = 53, figsize = (15, 15), show_info = ["split_gain"])
# plt.show()

# print("Plotting 54th tree with graphviz...")
# graph = lgb.create_tree_digraph(gbm, tree_index = 53, name = "Tree54")
# graph.render(view = True)
