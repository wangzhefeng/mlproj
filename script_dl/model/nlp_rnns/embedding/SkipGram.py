# -*- coding: utf-8 -*-


# ***************************************************
# * File        : skip_gram.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-25
# * Version     : 0.1.032519
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


class SkipGram(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int) -> None:
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = EMBED_DIMENSION,
            max_norm = EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features = EMBED_DIMENSION,
            out_features = vocab_size,
        )
    
    def fowrad(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x









# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
