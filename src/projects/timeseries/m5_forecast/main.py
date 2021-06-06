# import packages
import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt
from statsmodels.robust import mad

import scipy
import statsmodels
# signal
from scipy import signal
import statsmodels.api as sm
from scipy.signal import butter, deconvolve
# model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from fbprophet import Prophet

import warnings
warnings.filterwarnings("ignore")

from utils import fig_2_add_trace


# ==============================================================================
# config
# ==============================================================================
root = "."
data_path = os.path.join(root, "data")
TARGET = "sales"         # target
END_TRAIN = 1913         # last day in train dataset
MAIN_INDEX = ["id", "d"] # identify item by these columns

# ==============================================================================
# data
# ==============================================================================
sales_train_val = pd.read_csv(os.path.join(data_path, "sales_train_validation.csv"))
selling_prices = pd.read_csv(os.path.join(data_path, "sell_prices.csv"))
calendar = pd.read_csv(os.path.join(data_path, "calendar.csv"))
sales_train_val.sample(6)
selling_prices.sample(6)
calendar.sample(6)

# ==============================================================================
# EDA
# ==============================================================================
ids = sorted(list(set(sales_train_val["id"])))
d_cols = [c for c in sales_train_val.columns if "d_" in c]
x1 = sales_train_val.loc[sales_train_val["id"] == ids[1]].set_index("id")[d_cols].values[0]
x2 = sales_train_val.loc[sales_train_val["id"] == ids[2]].set_index("id")[d_cols].values[0]
x3 = sales_train_val.loc[sales_train_val["id"] == ids[3]].set_index("id")[d_cols].values[0]
x4 = sales_train_val.loc[sales_train_val["id"] == ids[4]].set_index("id")[d_cols].values[0]
print(x1)
print(x2)
print(x3)
print(x4)
fig = make_subplots(rows = 1, cols = 4)
fig_2_add_trace(fig, x_start = 0, x_end = len(x1), y = x1, mode = "lines", name = "First sample", color = "mediumseagreen", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x2), y = x2, mode = "lines", name = "Second sample", color = "violet", row = 1, col = 2)
fig_2_add_trace(fig, x_start = 0, x_end = len(x3), y = x3, mode = "lines", name = "Third sample", color = "dodgerblue", row = 1, col = 3)
fig_2_add_trace(fig, x_start = 0, x_end = len(x4), y = x4, mode = "lines", name = "Forth sample", color = "pink", row = 1, col = 4)
fig.update_layout(height = 600, width = 1600, title_text = "Sample sales")
fig.show()


#ids[1],ids[2],ids[3] are random samples , you can choose any number
x_1 = sales_train_val.loc[sales_train_val['id'] == ids[1]].set_index('id')[d_cols].values[0][:90]
x_2 = sales_train_val.loc[sales_train_val['id'] == ids[2]].set_index('id')[d_cols].values[0][0:90]
x_3 = sales_train_val.loc[sales_train_val['id'] == ids[3]].set_index('id')[d_cols].values[0][0:90]
x_4 = sales_train_val.loc[sales_train_val['id'] == ids[4]].set_index('id')[d_cols].values[0][0:90]
print(x_1)
print(x_2)
print(x_3)
print(x_4)
fig = make_subplots(rows=2, cols=2)
fig_2_add_trace(fig, x_start = 0, x_end = len(x_1), y = x_1, mode = "lines+markers", name = "First sample", color = "mediumseagreen", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x_2), y = x_2, mode = "lines+markers", name = "Second sample", color = "violet", row = 1, col = 2)
fig_2_add_trace(fig, x_start = 0, x_end = len(x_3), y = x_3, mode = "lines+markers", name = "Third sample", color = "dodgerblue", row = 1, col = 3)
fig_2_add_trace(fig, x_start = 0, x_end = len(x_4), y = x_4, mode = "lines+markers", name = "Forth sample", color = "pink", row = 1, col = 4)
fig.update_layout(height = 600, width = 1600, title_text = "Sample sales snippets")
fig.show()


# ==============================================================================
# Wavelet Denoising
# ==============================================================================
def maddest(d, axis = None):
    result = np.mean(np.absolute(d - np.mean(d, axis)), axis)

    return result


def denoise_signal(x, wavelet = "db4", level = 1):
    coeff = pywt.wavedec(x, wavelet, mode = "per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value = uthresh, mode = "hard") for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode = "per")

y_w1 = denoise_signal(x1)
y_w2 = denoise_signal(x2)
y_w3 = denoise_signal(x3)
y_w4 = denoise_signal(x4)

fig_2_add_trace(fig, x_start = 0, x_end = len(x1), y = x1, mode = "lines+markers", name = "First sample", color = "mediumseagreen", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x1), y = y_w1, mode = "lines+markers", name = "First sample", color = "black", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x2), y = x2, mode = "lines+markers", name = "Second sample", color = "violet", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x2), y = y_w2, mode = "lines+markers", name = "Second sample", color = "black", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x3), y = x3, mode = "lines+markers", name = "Third sample", color = "dodgerblue", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x3), y = y_w3, mode = "lines+markers", name = "Third sample", color = "black", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x4), y = x4, mode = "lines+markers", name = "Fourth sample", color = "pink", row = 4, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x4), y = y_w4, mode = "lines+markers", name = "Fourth sample", color = "black", row = 4, col = 1)
fig.update_layout(height = 1200, width = 800, title_text = "Sample sales snippets")
fig.show()

# ==============================================================================
# Average Smoothing
# ==============================================================================
def average_smoothing(signal, kernel_size = 3, stride = 1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start) * np.mean(signal[start:end]))
    
    return np.array(sample)

y_a1 = average_smoothing(x1)
y_a2 = average_smoothing(x2)
y_a3 = average_smoothing(x3)
y_a4 = average_smoothing(x4)

fig = make_subplots(rows=4, cols=1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x1), y = x1, mode = "lines+markers", name = "First sample", color = "mediumseagreen", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x1), y = y_a1, mode = "lines+markers", name = "First sample", color = "black", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x2), y = x2, mode = "lines+markers", name = "Second sample", color = "violet", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x2), y = y_a2, mode = "lines+markers", name = "Second sample", color = "black", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x3), y = x3, mode = "lines+markers", name = "Third sample", color = "dodgerblue", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x3), y = y_a3, mode = "lines+markers", name = "Third sample", color = "black", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x4), y = x4, mode = "lines+markers", name = "Fourth sample", color = "pink", row = 4, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = len(x4), y = y_a4, mode = "lines+markers", name = "Fourth sample", color = "black", row = 4, col = 1)
fig.update_layout(height=1200, width=800, title_text="Sample sales snippets")
fig.show()


fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(30, 20))
ax[0, 0].plot(x1, color='seagreen', marker='o') 
ax[0, 0].set_title('Original Sales', fontsize=24)
ax[0, 1].plot(y_a1, color='red', marker='.') 
ax[0, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[1, 0].plot(x2, color='seagreen', marker='o') 
ax[1, 0].set_title('Original Sales', fontsize=24)
ax[1, 1].plot(y_a2, color='red', marker='.') 
ax[1, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[2, 0].plot(x3, color='seagreen', marker='o') 
ax[2, 0].set_title('Original Sales', fontsize=24)
ax[2, 1].plot(y_a3, color='red', marker='.') 
ax[2, 1].set_title('After Wavelet Denoising', fontsize=24)

ax[3, 0].plot(x4, color='seagreen', marker='o') 
ax[3, 0].set_title('Original Sales', fontsize=24)
ax[3, 1].plot(y_a4, color='red', marker='.') 
ax[3, 1].set_title('After Wavelet Denoising', fontsize=24)
plt.show()



# ==============================================================================
# ARIMA
# ==============================================================================
# =========================
# train validation split
# =========================
train_dataset = sales_train_val[d_cols[-100:-30]]
val_dataste = sales_train_val[d_cols[-30:]]

fig = make_subplots(rows=3, cols=1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[0].values, mode = "lines", name = "Original signal", color = "dodgerblue", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[0].values, mode = "lines", name = "Denoised signal", color = "darkorange", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[1].values, mode = "lines", name = "Original signal", color = "dodgerblue", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[1].values, mode = "lines", name = "Denoised signal", color = "darkorange", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[2].values, mode = "lines", name = "Original signal", color = "dodgerblue", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[2].values, mode = "lines", name = "Denoised signal", color = "darkorange", row = 3, col = 1)
fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")
fig.show()




# =========================
# arima
# =========================
predictions = []
for row in tqdm(train_dataset[train_dataset.columns[-30:]].values[:3]):
    fit = sm.tsa.statespace.SARIMAX(row, seasonal_order = (0, 1, 1, 7)).fit()
    predictions.append(fit.forecast(30))

predictions = np.array(predictions).reshape((-1, 30))
error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3]) / len(predictions[0])


pred_1 = predictions[0]
pred_2 = predictions[1]
pred_3 = predictions[2]


fig = make_subplots(rows=3, cols=1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[0].values, mode = "lines", name = "Train", color = "dodgerblue", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[0].values, mode = "lines", name = "Val", color = "darkorange", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = pred_1, mode = "lines", name = "Pred", color = "seagreen", row = 1, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[1].values, mode = "lines", name = "Train", color = "dodgerblue", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[1].values, mode = "lines", name = "Val", color = "darkorange", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = pred_2, mode = "lines", name = "Pred", color = "seagreen", row = 2, col = 1)
fig_2_add_trace(fig, x_start = 0, x_end = 70, y = train_dataset.loc[2].values, mode = "lines", name = "Train", color = "dodgerblue", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = val_dataset.loc[2].values, mode = "lines", name = "Val", color = "darkorange", row = 3, col = 1)
fig_2_add_trace(fig, x_start = 70, x_end = 100, y = pred_3, mode = "lines", name = "Pred", color = "seagreen", row = 3, col = 1)
fig.update_layout(height=1200, width=800, title_text="ARIMA")
fig.show()
