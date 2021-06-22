from functools import reduce
import numpy as np


"""
对数据进行内存压缩
"""


def reduce_memory_usage(data, verbose = True):
    """
    reduce memory
    Args:
        data ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.
    """
    start_memory = data.memory_usage().sum() / 1024 ** 2
    print("Memory usage before optimization is: {:.4f} MB".format(start_memory))
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64",]
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in numerics:
            col_min = data[col].min()
            col_max = data[col].max()
            if str(col_type)[:3] == "int":
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
    end_memory = data.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.4f} MB".format(end_memory))
    print("Memory decreased by {:.1f}%".format(100 * (start_memory - end_memory) / start_memory))
    return data





if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
        "a": range(1, 100),
        "b": range(1, 100)
    })
    start_memory = df.memory_usage().sum() / 1024 ** 2
    print(start_memory)
    df = reduce_memory_usage(df)
