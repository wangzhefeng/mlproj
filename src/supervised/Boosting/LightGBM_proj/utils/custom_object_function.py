# -*- coding: utf-8 -*-


# ***************************************************
# * File        : custom_object_function.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-09
# * Version     : 0.1.080922
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import numpy as np


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def loglikelihood(pred, train_data):
    """
    self-defined objective function
    f(preds: array, train_data: Dataset) -> grad: array, hess: array
    log likelihood loss

    Args:
        pred ([type]): [description]
        train_data ([type]): [description]
    """
    labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    
    return grad, hess




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

