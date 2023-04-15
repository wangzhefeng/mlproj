# -*- coding: utf-8 -*-


# ***************************************************
# * File        : stop_words.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from nltk.corpus import stopwords

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# English stop words
stopwords_en = stopwords.words("english")
# TODO
# stopwords_zh = stopwords.words("")




# 测试代码 main 函数
def main():
    stopwords_en = stopwords.words("english")
    print(stopwords_en)

if __name__ == "__main__":
    main()
