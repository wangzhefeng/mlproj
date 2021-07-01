# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


# 数据
link_infos = pd.read.csv("../raw/gy_contest_link_info.txt", delimiter = ";", dtype = {"link_ID": object})
link_top = pd.read_csv("../raw/gy_contest_link_top.txt", delimiter = ";", dtype = {"link_ID": object})
df = pd.read_csv("../raw/quaterfinal_gy_cmp_training_traveltime.txt", delimiter = ";", dtype = {"link_ID": object})


# 探索性数据分析(EDA)

# 特征变换
df["travel_time"] = np.log1p(df["travel_time"])

# 数据平滑
def quantile_clip(group):
	# group.plot()
	group[group < group.quantile(0.05)] = group.quantile(0.05)
	group[group > group.quantile(0.95)] = group.quantile(0.95)
	# group.plot()
	plt.show()
	return
df["travel_time"] = df.groupby(["link_ID", "date"])["travel_time"].transform(quantile_clip)

# 缺失值补全
date_range = pd.date_range("2016-07-01 00:00:00", "2016-07-31 00:00:00", freq = "2min") \
	.append(pd.date_range("2017-04-01 00:00:00", "2016-07-31 00:00:00", freq = "2min"))
new_index = pd.MultiIndex.from_product([link_df["link_ID"].unique(), date_range],
									   names = ["link_ID", "time_interval_begin"])
df1 = pd.DateFrame(index = new_index).reset_index()
df3 = pd.merge(df1, df, on = ["link_ID", "time_interval_begin"], how = "left")