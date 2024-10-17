# -*- coding: utf-8 -*-

# ***************************************************
# * File        : adaboost.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-05
# * Version     : 0.1.040522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, 
    iris.target,
    test_size = 0.3,
    random_state = 27149
)


# ------------------------------
# model
# ------------------------------
ab = AdaBoostClassifier(n_estimators = 100)
ab.fit(X_train, y_train)
accuracy = ab.score(X_test, y_test)
print("Hold Out 分割数据集模型在测试集上的准确率为: %0.03f" % accuracy)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
