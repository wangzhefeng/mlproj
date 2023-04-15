# -*- coding: utf-8 -*-


# ***************************************************
# * File        : wikidata_cleaning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-15
# * Version     : 0.1.041521
# * Description : description
# * Link        : link
# * Requirement : python wikidata_cleaning.py input_file > output_file
# ***************************************************


# python libraries
import os
import sys

import re

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def remove_empty_paired_punc(in_str):
    """
    删除空的成对符号
    """
    in_str = in_str \
        .replace("（）", "") \
        .replace("《》", "") \
        .replace("【】", "") \
        .replace("[]", "")
    
    return in_str


def remove_html_tag(in_str):
    """
    删除多余的 html 标签
    """
    html_pattern = re.compile(r"<[^>]+>", re.S)

    return html_pattern.sub("", in_str)


def remove_control_chars(in_str):
    """
    删除不可见控制字符
    """
    control_chars = "".join(map(
        chr, 
        list(range(0, 32)) + 
        list(range(127, 160))
    ))
    control_chars = re.compile("[%s]" % re.escape(control_chars))

    return control_chars.sub("", in_str)




# 测试代码 main 函数
def main():
    with open(sys.argv[1], "r") as f_in:  # 输入文件
        for line in f_in.readlines():
            line = line.strip()
            if re.search(r"^(<doc id)|(</doc>)", line):  # 跳过文档 html 标签行
                print(line)
                continue
            line = remove_empty_paired_punc(line)
            line = remove_html_tag(line)
            line = remove_control_chars(line)
            print(line)

if __name__ == "__main__":
   main()
