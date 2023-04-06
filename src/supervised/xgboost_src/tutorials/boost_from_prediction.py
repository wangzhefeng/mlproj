# -*- coding: utf-8 -*-


# ***************************************************
# * File        : boost_from_prediction.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-07-20
# * Version     : 0.1.072000
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XGBOOST_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(XGBOOST_ROOT_DIR, "demo")
train_dir = os.path.join(CURRENT_DIR, '../data/agaricus.txt.train')
test_dir = os.path.join(CURRENT_DIR, '../data/agaricus.txt.test')


# data
dtrain = xgb.DMatrix(train_dir)
dtest = xgb.DMatrix(test_dir)

# model perference
watchlist = [
    (dtest, 'eval'), 
    (dtrain, 'train'),
]

# model
print('start running example to start from a initial prediction')
param = {
    'max_depth': 2, 
    'eta': 1, 
    'objective': 'binary:logistic',
}
bst = xgb.train(
    param, 
    dtrain, 
    1, 
    watchlist,
)
ptrain = bst.predict(dtrain, output_margin = True)
ptest = bst.predict(dtest, output_margin = True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)

print('this is result of running from initial prediction')
bst = xgb.train(
    param, 
    dtrain, 
    1, 
    watchlist,
)





# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

