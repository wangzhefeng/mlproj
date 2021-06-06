import numpy as np
import pandas as pd
import warnings
import itertools
import random
import statsmodels.api as sm
from fbprophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# read data
# df = pd.read_csv("../input/groceries-sales-data/Groceries_Sales_data.xlsx", parse_dates = [0])

# # data info
# print(df.head())
# print(df.tail())
# print(df.info())



# EDA



# HyperParmeter Tuning using ParameterGrid
# parameters
params_grid = {
    "seasonality_mode": ("multiplicative", "additive"),
    "changepoint_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
    "holidays_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_changepoints": [100, 150, 200],
}
grid = ParameterGrid(params_grid)
cnt = 0
for p in grid:
    cnt = cnt + 1
print("Total Possible Models: %s" % cnt)

# prophet model tuning
