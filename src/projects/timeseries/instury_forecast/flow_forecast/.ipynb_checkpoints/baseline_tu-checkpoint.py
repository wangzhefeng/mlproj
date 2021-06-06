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
data = path = os.path.join(os.path.join(root_path, "data/preprocessed_data"))


# -----------------------
# read data
# -----------------------
all_train_data = pd.read_csv(os.path.join(data, "all_train_data.csv"), header = 0, index_col = None)
test_data_month1 = pd.read_csv(os.path.join(data, "test_data_month1.csv"), header = 0, index_col = None)
test_data_month7 = pd.read_csv(os.path.join(data, "test_data_month7.csv"), header = 0, index_col = None)
test_data_month10 = pd.read_csv(os.path.join(data, "test_data_month10.csv"), header = 0, index_col = None)
print("-" * 50)
print("all_train_data info:")
print("-" * 50)
print(all_train_data.head())
print(all_train_data.info())


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
# extra tree regressor
# -----------------------
reg = ExtraTreesRegressor(
    bootstrap = False,
    ccp_alpha = 0.0, 
    criterion = "mse",
    max_depth = None,
    max_features = "auto",
    max_leaf_nodes = None,
    max_samples = None,
    min_impurity_decrease = 0.0,
    min_samples_split = 2,
    min_weight_fraction_leaf = 0.0,
    n_estimators = 100,
    n_jobs = -1,
    oob_score = False,
    random_state = 8603,
    verbose = 0,
    warm_start = False
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("-" * 50)
print("model performance:")
print("-" * 50)
print("Mae:", mean_absolute_error(y_pred, y_test))
print("Mse:", mean_squared_error(y_pred, y_test))
