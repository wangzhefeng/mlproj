# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ltp_utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041522
# * Description : description
# * Link        : https://github.com/HIT-SCIR/ltp/blob/main/python/interface/README.md
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from ltp import LTP


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 默认加载 Small 模型
ltp = LTP()

# 分词
segment, hidden = ltp.seg(["南京市长江大桥。"])
# print(segment)

print(dir(ltp))


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
