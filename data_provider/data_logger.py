# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_logger.py
# * Author      : Zhefeng Wang
<<<<<<<< HEAD:utils/data_logger.py
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-17
# * Version     : 0.1.101715
# * Description : 数据日志查看
========
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-19
# * Version     : 1.0.091900
# * Description : description
>>>>>>>> 6a4fe4a8f1ca2bb97e0748fe9b4b4310683e6dab:data_provider/data_logger.py
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

<<<<<<<< HEAD:utils/data_logger.py
========
__all__ = [
    "timeseries_info_logging",
    "df_print",
    "df_logging",
]

>>>>>>>> 6a4fe4a8f1ca2bb97e0748fe9b4b4310683e6dab:data_provider/data_logger.py
# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import logging

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
logger = logging.getLogger()


def df_print(data: pd.DataFrame, data_name: str):
    """
    查看数据结构、内容信息
    """
    print("=" * 40)
    print(f"{data_name}.head()")
    print("=" * 40)
    print(data.head())
    print("=" * 40)
    print(f"{data_name}.tail()")
    print("=" * 40)
    print(data.tail())
    print("=" * 40)
    print(f"{data_name}.info()")
    print("=" * 40)
    print(data.info())


def df_logging(data: pd.DataFrame, data_name: str):
    """
    查看数据结构、内容信息
    """
    logger.info("=" * 40)
    logger.info(f"{data_name}.head()")
    logger.info("=" * 40)
    logger.info(data.head())
    logger.info("=" * 40)
    logger.info(f"{data_name}.tail()")
    logger.info("=" * 40)
    logger.info(data.tail())
    logger.info("=" * 40)
    logger.info(f"{data_name}.info()")
    logger.info("=" * 40)
    logger.info(data.info())


def timeseries_info_logging(ts, data_name):
    pass


<<<<<<<< HEAD:utils/data_logger.py


========
>>>>>>>> 6a4fe4a8f1ca2bb97e0748fe9b4b4310683e6dab:data_provider/data_logger.py
# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
