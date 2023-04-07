# -*- coding: utf-8 -*-


# ***************************************************
# * File        : cbow.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CBOW(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, voacab_size: int) -> None:
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings = voacab_size,
            embedding_dim = EMBED_DIMENSION,
            max_norm = EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features = EMBED_DIMENSION, 
            out_features = voacab_size
        )

    def forward(self, inputs_):
        x = self.embedding(inputs_)
        x = x.mean(axis = 1)
        x = self.linear(x)
        return x








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
