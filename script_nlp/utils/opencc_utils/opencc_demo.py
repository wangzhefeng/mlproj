# -*- coding: utf-8 -*-


# ***************************************************
# * File        : opencc_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-22
# * Version     : 0.1.032214
# * Description : description
# * Link        : link
# * Requirement : pip install opencc
# *               python convert_t2s.py input_file > output_file
# ***************************************************


# python libraries
import os
import sys

import opencc

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]




# 测试代码 main 函数
def main():
    # 繁体
    data_old = u'Open Chinese Convert（OpenCC）是一個開源的中文簡繁轉換項目，致力於製作高質量的基於統計預料的簡繁轉換詞庫。還提供函數庫(libopencc)、命令行簡繁轉換工具、人工校對工具、詞典生成程序、在線轉換服務及圖形用戶界面'
    # 日语
    data_japan = u"プレバト"
    # 转换
    converter = opencc.OpenCC("jp2t.json")
    data_new = converter.convert(data_japan)
    print(data_new)

if __name__ == "__main__":
    main()
