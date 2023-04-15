# -*- coding: utf-8 -*-


# ***************************************************
# * File        : datasets_utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041521
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from pprint import pprint
from datasets import list_datasets, load_dataset
import evaluate

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


datasets_list = list_datasets()
print(len(datasets_list))



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
