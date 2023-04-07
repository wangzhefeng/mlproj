# -*- coding: utf-8 -*-


# ***************************************************
# * File        : TextClassifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-03
# * Version     : 0.1.040319
# * Description : description
# * Link        : https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-simple-guide-to-text-classification#2
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TextClassifier(nn.Module):

    def __init__(self, vocab) -> None:
        super(TextClassifier, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(len(vocab), 128),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.linear3 = nn.Linear(64, 4)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
