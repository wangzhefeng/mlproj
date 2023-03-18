# -*- coding: utf-8 -*-


# ***************************************************
# * File        : base_walkthrough.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-19
# * Version     : 0.1.071922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import pickle

import numpy as np
from sklearn.datasets import load_svmlight_file
import xgboost as xgb


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XGBOOST_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(XGBOOST_ROOT_DIR, "demo")
train_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.train")
test_dir = os.path.join(CURRENT_DIR, "../data/agaricus.txt.test")


# ----------------
# data
# ----------------
# train data
X, y = load_svmlight_file(train_dir)
dtrain = xgb.DMatrix(X, y)
# validation set
X_test, y_test = load_svmlight_file(test_dir)
dtest = xgb.DMatrix(X_test, y_test)

# ----------------
# params
# ----------------
param = {
    "max_depth": 2,
    "eta": 1,
    "objective": "binary:logistic",
}

# ----------------
# perference view
# ----------------
watchlist = [
    (dtest, "eval"),
    (dtrain, "train"),
]

# ----------------
# model train
# ----------------
num_round = 2
bst = xgb.train(
    param,
    dtrain, 
    num_boost_round = num_round,
    evals = watchlist
)

# ----------------
# model predict
# ----------------
preds = bst.predict(dtest)
labels = dtest.get_label()

print(
    f"error={sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))}"
)

# ----------------
# model save
# ----------------
bst.save_model("model-0.json")

# dump model
bst.dump_model("dump.raw.txt")

# dump model with feature map
bst.dump_model("dump.nice.txt", os.path.join(DEMO_DIR, "data/featmap.txt"))

# save dmatrix into binary buffer
dtest.save_binary("dtest.dmatrix") 

# save model
bst.save_model("model-1.json")

# pickle booster
pks = pickle.dumps(bst)

# ----------------
# load model and data
# ----------------
bst2 = xgb.Booster(model_file = "model-1.json")
dtest2 = xgb.DMatrix("dtest.dmatrix")
preds2 = bst2.predict(dtest2)
assert np.sum(np.abs(preds2 - preds)) == 0


bst3 = pickle.loads(pks)
dtest2 = xgb.DMatrix("dtest.dmatrix")
preds3 = bst3.predict(dtest2)
assert np.sum(np.abs(preds3 - preds)) == 0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

