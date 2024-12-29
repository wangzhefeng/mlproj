# -*- coding: utf-8 -*-

# ***************************************************
# * File        : baggging.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# bagging ensemble of different classifiers
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
X_train, y_train = None, None

# model fusion
clf = BaggingClassifier(
    base_estimator = SVC(),
    n_estimators = 10, 
    random_state = 0)
clf.fit(X_train,y_train)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
