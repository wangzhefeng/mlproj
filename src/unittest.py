# -*- coding: utf-8 -*-


import pandas as pd
# 缺失值补全
date_range = pd.date_range("2016-07-01 00:00:00", "2016-07-31 00:00:00", freq = "2min") \
	.append(pd.date_range("2017-04-01 00:00:00", "2016-07-31 00:00:00", freq = "2min"))


with pd.option_context("display.max_rows", None):
	print(date_range)


