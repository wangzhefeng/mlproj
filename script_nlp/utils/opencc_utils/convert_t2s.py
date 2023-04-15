# -*- coding: utf-8 -*-


# ***************************************************
# * File        : convert_t2s.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041521
# * Description : description
# * Link        : link
# * Requirement : python convert_t2s.py input_file > output_file
# ***************************************************


# python libraries
import os
import sys

import opencc

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]




# 测试代码 main 函数
def main():
    converter = opencc.OpenCC("t2s.json")  # 载入繁简体转换配置文件
    with open(sys.argv[1], "r") as f_in:
        for line in f_in.readlines():
            line = line.strip()
            line_t2s = converter.convert(line)
            print(line_t2s)


if __name__ == "__main__":
    main()
