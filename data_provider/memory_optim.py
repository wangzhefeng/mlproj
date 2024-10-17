# -*- coding: utf-8 -*-

# ***************************************************
# * File        : memory_optim.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-19
# * Version     : 1.0.091900
# * Description : 内存优化方法：
# *               1.查看数据列和行，读取需要的数据
# *               2.查看数据类型，进行类型转换
# *               3.分批次或利用磁盘，处理数据
# * Link        : https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247501326&idx=1&sn=a6c3757625cfeb3e897c16a07b435931&chksm=96c7ebcba1b062dd4cca34b31ad21a25d59b75d341152f1052437a46f12d35d2d7e9a0f4fd66&cur_album_id=1364202321906941952&scene=189#wechat_redirect
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import psutil
from typing import List, Dict

import numpy as np
import pandas as pd
import sparse

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def cpu_stats():
    """
    当前进程内存使用统计
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0 ** 30
    cpu_stats_info = f"memory GB: {str(np.round(memory_use, 2))}"
    
    return cpu_stats_info


def pandas_data_memory(df: pd.DataFrame) -> None:
    """
    对于 pandas 读取的数据，查看内存使用信息
    """
    # 整体内存使用
    df_info = df.info(memory_usage = "deep")
    print(f"df memory usage: {df_info}")
    # 每列内存使用
    column_info = df.memory_usage()
    print(f"df memory usage per column: {column_info}")


def numpy_data_memory(arr: np.ndarray) -> None:
    """
    Numpy 内存优化
        - 在 Numpy 支持多种数据类型，不同类型数据的内存占用相差很大
        - 对于数据类型，可以根据矩阵的元素范围进行设置
    Numpy data type: https://numpy.org/devdocs/user/basics.types.html
    """
    arr_info = arr.nbytes
    print(f"array memory usage: {arr_info}")


def sparse_matrix_optim(matirx_array: np.ndarray) -> np.ndarray:
    """
    如果矩阵中数据是稀疏的情况，可以考虑稀疏矩阵。LGB和XGB支持稀疏矩阵参与训练
    """
    print(f"Non matrix nbytes: {matirx_array.nbytes}") 
    sparse_matrix_array = sparse.COO(matirx_array)
    print(f"sparse matrix nbytes: {sparse_matrix_array.nbytes}")

    return sparse_matrix_array


def pandas_data_read(path, 
                     usecols: List[str], 
                     catecols: List[str], 
                     cols_dtype: Dict, 
                     batch_size: int = None) -> pd.DataFrame:
    """
    Pandas内存优化：
    
    1. 分批读取
        - 如果数据文件非常大，可以在读取时分批次读取，通过设置 `chunksize` 来控制批大小
    2. 选择读取部分列
    3. 提前设置列类型
    4. 将类别列设为 `category` 类型
        - 此操作对于类别列压缩非常有效，压缩比很大。
          同时在设置为 `category` 类型后，
          LightGBM 可以视为类别类型训练
    """
    # 数据读取
    df = pd.read_csv(path, usecols = usecols, dtype = cols_dtype, chunksize = batch_size)
    # 分批处理数据
    if batch_size:
        for chunk in df:
            pass
    # 将类别设置为 category 类型
    for catecol in catecols:
        df[catecol] = df[catecol].astype("category")

    # 内存使用信息
    pandas_data_memory(df)

    return df


def pandas_reduce_memory(data: pd.DataFrame) -> pd.DataFrame:
    """
    pandas DataFrame reduce memory

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # memory usage
    start_memory_usage = data.memory_usage().sum() / 1024 ** 2
    print("# " + "-" * 70)
    print(f"# Memory usage before optimization is: {start_memory_usage:.4f} MB")
    print("# " + "-" * 50)
    # TODO numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    # keeps track of columns that have missing values filled in
    na_list = []
    for col in data.columns:
        col_type = data[col].dtype
        if col_type != object:  # exclude strings or if col_type in numerics:
            print(f"column: {col}, dtype before: {col_type}")
            # make variables for Int, max and min
            is_int = False
            col_min = data[col].min()
            col_max = data[col].max()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(data[col]).all():
                na_list.append(col)
                data[col].fillna(col_min - 1, inplace = True)
            # test if column can be converted to an integer
            as_int = data[col].fillna(0).astype(np.int64)
            result = (data[col] - as_int).sum()
            if result > -0.01 and result < 0.01:
                is_int = True
            # type convert
            if is_int:
                if col_min >= 0:
                    if col_max < 255:
                        data[col] = data[col].astype(np.uint8)
                    elif col_max < 65535:
                        data[col] = data[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.uint64)
                else:
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)
            else:
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
            # print new column type
            print(f"column: {col}, dtype after: {data[col].dtype}")
            print("*" * 35)
    # Print final result
    end_memory_usage = data.memory_usage().sum() / 1024 ** 2
    print("# " + "-" * 50)
    print(f"# Memory usage after optimization is: {end_memory_usage:.4f} MB")
    print("# " + "-" * 50)
    print(f"# Memory decreased by {100 * (start_memory_usage - end_memory_usage) / start_memory_usage:.1f}%")
    print("# " + "-" * 70)
    
    return data, na_list




# 测试代码 main 函数
def main():
    # reduce_memory_usage
    import pandas as pd
    df = pd.DataFrame({
        "a": range(1, 100),
        "b": range(1, 100)
    })
    df, na_list = pandas_reduce_memory(df)

    # cpu memory usage
    res = cpu_stats()
    print(res)
    
    # psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    print(py.memory_info())

if __name__ == "__main__":
    main()
