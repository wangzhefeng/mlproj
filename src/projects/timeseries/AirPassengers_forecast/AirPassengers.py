# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.pyplot import rcParams
rcParams["figure.figsize"] = 15, 6


# ######################################
# 读取数据为 DataFrame
# ######################################
dateparse = lambda dates: pd.to_datetime(dates, format = "%Y-%m")
data = pd.read_csv("../data/AirPassengers/AirPassengers.csv",
				   parse_dates = ["Month"],
				   index_col = "Month", 
				   date_parser = dateparse)
print(data.shape)
print("-" * 25)
print(data.head())
print("-" * 25)
print(data.dtypes)
print("-" * 25)
print(data.index)
print("-" * 25)


# #####################################
# 将数据转换为 Series
# #####################################
ts = data["#Passengers"]
print(ts.head())
print()
print(ts["1949-01-01"])
print()
print(ts[datetime.datetime(1949, 1, 1)])
print()
print(ts["1949-01-01":"1949-05-01"])
print()
print(ts[:"1949-05-01"])
print()
print(ts["1949"])

# #####################################
# 时间序列分析——检验时间序列的平稳性
# #####################################
plt.plot(ts)
plt.show()













