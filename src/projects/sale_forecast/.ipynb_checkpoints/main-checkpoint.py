import pandas as pd



start = "2013-01-01 00:00:00"
end = "2017-12-31 23:00:00"
start_2018_01 = "2018-01-01 00:00:00"
end_2018_01 = "2018-01-24 23:00:00"
start_2018_07 = "2018-07-01 00:00:00"
end_2018_07 = "2018-07-24 23:00:00"
start_2018_10 = "2018-10-01 00:00:00"
end_2018_10 = "2018-10-24 23:00:00"

datetime_index_one = pd.date_range(start, end, freq='3h')
datetime_index_two = pd.date_range(start_2018_01, end_2018_01, freq='3h')
datetime_index_three = pd.date_range(start_2018_07, end_2018_07, freq='3h')
datetime_index_four = pd.date_range(start_2018_10, end_2018_10, freq='3h')
print(datetime_index_one)
print(datetime_index_two)
print(datetime_index_three)
print(datetime_index_four)
L = [x.strftime('%Y-%m-%d %H:%M:%S') for x in list()]
print(L)