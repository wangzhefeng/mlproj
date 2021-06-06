import pandas as pd
import numpy as np
from glob import glob
from multiprocessing import Pool


def get_statistics_features(c, data):
    
    columns = data.columns.to_list()
    for col in columns[1:]:
        # for fea in feature_list:
            # exec("c.update({'%s_%s': data[%s].%s()})" % (col, fea, col, fea))
        c.update({f"{col}_max": data[col].max()})
        c.update({f"{col}_min": data[col].min()})
        c.update({f"{col}_max-min": data[col].max() - data[col].min()})
        c.update({f"{col}_std": data[col].std()})
        c.update({f"{col}_skew": data[col].skew()})
        c.update({f"{col}_mean": data[col].mean()})

    return c


def get_single_feature(path):
    print("process path", path)
    data = pd.read_csv(path)
    ID = path.split("_")[-1].split(".")[0]
    c = {
        "Id": ID,
        "SampleTime": data["SampleTime"].max()
    }
    c = get_statistics_features(c, data)
    df = pd.DataFrame(c, index=[0])
    # print(df.head())
    return df


def get_feature_together(cpu, func, data_paths):
    rst = []
    pool = Pool(cpu)
    for data_path in data_paths:
        rst.append(pool.apply_async(func, args = (data_path,  )))
    pool.close()
    pool.join()

    rst = [i.get() for i in rst]
    tv_features = rst[0]
    for i in rst[1:]:
        tv_features = pd.concat([tv_features, i], axis = 0)
    
    return tv_features



if __name__ == "__main__":
    data_paths = glob('/Users//data/Train/传感器高频数据/*.csv')
    sensor_train = get_feature_together(4, get_single_feature, data_paths)
    sensor_train.to_csv("sensor_test.csv", index = None)
