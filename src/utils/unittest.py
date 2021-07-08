# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# import pdb
# def make_break():
#     pdb.set_trace()
#     return "I don't habe time."
# print(make_break())





def create_cv(x_train_month, x_train, y_train, n):
    """
    自定义交叉验证(月)
    Args:
        x_train_month ([type]): [description]
        x_train ([type]): [description]
        y_train ([type]): [description]
        n ([type]): [description]
    """
    def data2list(lst):
        """
        #TODO
        """
        ret = []
        for i in lst:
            ret += i
        return ret
    
    groups = x_train.groupby(x_train_month).groups
    sorted_groups = [value.tolist() for (key, value) in sorted(groups.items())]
    cv = [(np.array(data2list(sorted_groups[i:i+n])), np.array(sorted_groups[i+n])) for i in range(len(sorted_groups) - n)]
    return cv






if __name__ == "__main__":
    x_train = pd.DataFrame(
        list(range(100)), 
        columns = ["col0"]
    )
    y_train = pd.DataFrame(
        [np.random.randint(0, 2) for i in range(100)], 
        columns = ["y"]
    )
    x_train_month = ['2018-01'] * 20 + \
        ['2018-02'] * 20 + \
        ['2018-03'] * 20 + \
        ['2018-04'] * 20 + \
        ['2018-05'] * 20
    n = 3 # 3个月训练，1个月验证
    cv = create_cv(x_train_month, x_train, y_train, n)
    print(cv)
    
    # 搭配 GridSearchCV使用
    param_test = {
        "max_depth": list(range(5, 12, 2))
    }
    grid_search = GridSearchCV(
        estimator = XGBClassifier(),
        param_grid = param_test,
        cv = cv
    )






# [
#     (
#         array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#             51, 52, 53, 54, 55, 56, 57, 58, 59
#         ]), 
#         array([
#            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
#         ])
#     ), 
#     (
#         array([
#             20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
#             37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
#             54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
#             71, 72, 73, 74, 75, 76, 77, 78, 79
#         ]), 
#        array([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
#     )