# -*- coding: utf-8 -*-


# ***************************************************
# * File        : EmbeddingClassifier.py
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

from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class EmbeddingClassifier(nn.Module):

    def __init__(self, max_words, embed_len, target_classes) -> None:
        super(EmbeddingClassifier, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(max_words * embed_len, 256),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.linear4 = nn.Linear(64, len(target_classes))
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
