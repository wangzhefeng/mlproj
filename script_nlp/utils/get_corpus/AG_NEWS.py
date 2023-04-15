# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AG_NEWS.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-03
# * Version     : 0.1.040319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys


import torch
from torch.utils.data import DataLoader

import torchtext


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data
train_dataset, test_dataset = torchtext.datasets.AG_NEWS()
target_classes = ["World", "Sports", "Business", "Sci/Tec"]









# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
