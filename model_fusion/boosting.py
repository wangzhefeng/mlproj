# -*- coding: utf-8 -*-

# ***************************************************
# * File        : boosting.py
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# data
x_train = None
y_train = None

# model fusion
dt = DecisionTreeClassifier(max_depth = 2, random_state = 0)
adc = AdaBoostClassifier(
    base_estimator = dt, 
    n_estimators = 7, 
    learning_rate = 0.1, 
    random_state = 0
)
adc.fit(x_train, y_train)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
