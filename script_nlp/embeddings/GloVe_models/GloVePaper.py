# -*- coding: utf-8 -*-


# ***************************************************
# * File        : GloVePaper.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032804
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


class GloVe(nn.Module):

    def __init__(self, vocab_size, embed_size) -> None:
        super(GloVe).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 声明 v 和 w 为 Embedding 向量
        self.v = nn.Embedding(vocab_size, embed_size)
        self.w = nn.Embedding(vocab_size, embed_size)
        self.bias_v = nn.Embedding(vocab_size, 1)
        self.bias_w = nn.Embedding(vocab_size, 1)
        # 随机初始化参数
        init_range = 0.5 / self.embed_size
        self.v.weight.data.uniform_(-init_range, init_range)
        self.w.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, i, j, co_occur, weight):
        # 根据目标函数计算 Loss 值
        vi = self.v(i)  # 分别根据索引i和j取出对应的词向量和偏差值
        wj = self.w(j)
        bi = self.bias_v(i)
        bj = self.bias_w(j)
        similarity = torch.sum(torch.mul(vi, wj), dim = 1)
        loss = similarity + bi + bj - torch.log(co_occur)
        loss = 0.5 * weight * loss * loss
        out = loss.sum().mean()

        return out
    
    def gloveMatrix(self):
        """
        获得词向量，这里把两个向量相加作为最后的词向量
        """
        self.v.weight.data.numpy() + self.w.weight.data.numpy()



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
