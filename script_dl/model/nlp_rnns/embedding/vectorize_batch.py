# -*- coding: utf-8 -*-


# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032805
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# hyper parameters
max_words = 50
embed_len = 300


# ------------------------------
# 分词器
# ------------------------------
tokenizer = get_tokenizer("basic_english")

# ------------------------------
# pre-train model
# ------------------------------
global_vectors = GloVe(name = "840B", dim = embed_len)


# ------------------------------
# data preprocessing
# ------------------------------
def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    # 分词
    X = [tokenizer(x) for x in X]
    # 填充空字符串(pad empty string)
    X = [
        tokens + [""] * (max_words - len(tokens)) 
        if len(tokens) < max_words
        else tokens[:max_words] 
        for tokens in X
    ]
    # 将输入转换为 tensor
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        # embedding
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    
    return X_tensor.reshape(len(batch), -1), torch.tensor(Y) - 1


def vectorize_avg_batch(batch):
    Y, X = list(zip(*batch))
    # 分词
    X = [tokenizer(x) for x in X]
    # 填充空字符串(pad empty string)
    X = [
        tokens + [""] * (max_words - len(tokens)) 
        if len(tokens) < max_words
        else tokens[:max_words] 
        for tokens in X
    ]
    # 将输入转换为 tensor
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        # embedding
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    
    return X_tensor.mean(dim = 1), torch.tensor(Y) - 1


def vectorize_sum_batch(batch):
    Y, X = list(zip(*batch))
    # 分词
    X = [tokenizer(x) for x in X]
    # 填充空字符串(pad empty string)
    X = [
        tokens + [""] * (max_words - len(tokens)) 
        if len(tokens) < max_words
        else tokens[:max_words] 
        for tokens in X
    ]
    # 将输入转换为 tensor
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        # embedding
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    
    return X_tensor.sum(dim = 1), torch.tensor(Y) - 1




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
