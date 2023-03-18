# -*- coding: utf-8 -*-


# ***************************************************
# * File        : custom_metrics.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-09
# * Version     : 0.1.080923
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


def rmsle(y_true, y_pred):
    """
    self-defined eval metric
    f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    Root Mean Squared Logarithmic Error (RMSLE)

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
    """
    rmsle_result = np.sqrt(
        np.mean(
            np.power(np.log1p(y_pred) - np.log1p(y_true), 2)
        )
    )
    
    return 'RMSLE', rmsle_result, False


def rae(y_true, y_pred):
    """
    self-defined eval metric
    f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    Relative Absolute Error (RAE)

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    rae_result = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true))

    return 'RAE', rae_result, False



def binary_error(preds, train_data):
    """
    # self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # binary error

    # NOTE: when you do customized loss function, the default prediction value is margin
    # This may make built-in evalution metric calculate wrong results
    # For example, we are doing log likelihood loss, the prediction is score before logistic transformation
    # Keep this in mind when you use the customization

    Args:
        preds ([type]): [description]
        train_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False



def accuracy(preds, train_data):
    """
    # another self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # accuracy
    # NOTE: when you do customized loss function, the default prediction value is margin
    # This may make built-in evalution metric calculate wrong results
    # For example, we are doing log likelihood loss, the prediction is score before logistic transformation
    # Keep this in mind when you use the customization

    Args:
        preds ([type]): [description]
        train_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

