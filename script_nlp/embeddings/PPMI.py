# -*- coding: utf-8 -*-


# ***************************************************
# * File        : PPMI.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import pandas as pd
from collections import defaultdict

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# TODO
def CoOccurrenceMatrix(sentences, window_size):
    """
    共现矩阵
    """
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1
    # formulate the dictionary into dataframe
    vocab = sorted(vocab)  # sort vocab
    df = pd.DataFrame(
        data = np.zeros((len(vocab), len(vocab)), dtype = np.int16),
        index = vocab,
        columns = vocab
    )
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    
    return df


def PIM(M, positive = True):
    """
    点互信息(Pointwise Mutual Infomation, PMI)
    PPMI(Positive PMI) = max(PMI(w, c), 0)

    Args:
        M (_type_): 共现矩阵
        positive (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    col_totals = M.sum(axis = 0)  # 按列求和
    row_totals = M.sum(axis = 1)  # 按行求和
    total = col_totals.sum()  # 总频次
    expected = np.outer(row_totals, col_totals) / total  # 获得每个元素的分子
    M = M / expected
    with np.errstate(divide = "ignore"):  # 不显示 log(0) 的警告
        M = np.log(M)
    M[np.isinf(M)] = 0.0  # 将 log(0) 设置为 0
    if positive:
        M[M < 0] = 0.0

    return M





# 测试代码 main 函数
def main():
    # text = [
    #     ["我", " 喜欢", "自然", "语言", "处理", "。"],
    #     ["我", "爱", "深度", "学习", "。"],
    #     ["我", "喜欢", "机器", "学习", "。"]
    # ]
    # co_matrix = CoOccurrenceMatrix(text, 10)
    # print(co_matrix)
    pass

if __name__ == "__main__":
    main()
