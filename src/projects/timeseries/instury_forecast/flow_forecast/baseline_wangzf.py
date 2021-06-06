import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")


# -----------------------
# data path
# -----------------------
root_path = "/Users/zfwang/project/machinelearning/timeseries/src/forecast_instury"
data_path = os.path.join(root_path, "data/preprocessed_data")
result_path = os.path.join(root_path, "result")

# -----------------------
# read data
# -----------------------
all_train_data = pd.read_csv(os.path.join(data_path, "all_train_data.csv"), header = 0, index_col = None)
test_data_month1 = pd.read_csv(os.path.join(data_path, "test_data_month1.csv"), header = 0, index_col = None)
test_data_month7 = pd.read_csv(os.path.join(data_path, "test_data_month7.csv"), header = 0, index_col = None)
test_data_month10 = pd.read_csv(os.path.join(data_path, "test_data_month10.csv"), header = 0, index_col = None)
print("-" * 50)
print("all_train_data info:")
print("-" * 50)
print(all_train_data.head())
print(all_train_data.info())

# reindex datetime
start_2013_2017 = "2013-03-11 00:00:00"
end_2013_2017 = "2017-12-31 23:00:00"
start_2018_01 = "2018-01-01 00:00:00"
end_2018_01 = "2018-01-24 23:00:00"
start_2018_07 = "2018-07-01 00:00:00"
end_2018_07 = "2018-07-24 23:00:00"
start_2018_10 = "2018-10-01 00:00:00"
end_2018_10 = "2018-10-24 23:00:00"
datetime_index_one = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in list(pd.date_range(start_2013_2017, end_2013_2017, freq='3h'))]
datetime_index_two = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in list(pd.date_range(start_2018_01, end_2018_01, freq='3h'))]
datetime_index_three = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in list(pd.date_range(start_2018_07, end_2018_07, freq='3h'))]
datetime_index_four = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in list(pd.date_range(start_2018_10, end_2018_10, freq='3h'))]
datetime_index = datetime_index_one + datetime_index_two + datetime_index_three + datetime_index_four
df = pd.DataFrame(data = datetime_index, index = None, columns = ["dt"])
df = df.merge(all_train_data, left_on = "dt", right_on = "TimeStamp", how = "left")
print("-" * 50)
print("new df info:")
print("-" * 50)
print(df.shape)
print(df.info())




# -----------------------
# split data
# -----------------------
X = all_train_data.loc[:, "R1":"D5"]
y = all_train_data.loc[:, "label":]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2020)

print("-" * 50)
print("train data shape:")
print("-" * 50)
print("The Shape of X_train:", X_train.shape)
print("The Shape of X_test:", X_test.shape)
print("The Shape of y_train:", y_train.shape)
print("The Shape of y_test:", y_test.shape)


# -----------------------
# model training
# -----------------------
print("-" * 50)
print("model performance:")
print("-" * 50)
# print("Mae:", mean_absolute_error(y_pred, y_test))
# print("Mse:", mean_squared_error(y_pred, y_test))
