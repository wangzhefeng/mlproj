# -*- coding: utf-8 -*-


# ***************************************************
# * File        : DeepAR.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-10
# * Version     : 0.1.041023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.util import to_pandas
from gluonts.mx import DeepAREstimator, Trainer


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class DeepAR(object):

    def __init__(self) -> None:
        pass

    def train(self, train_set):
        # model
        deepar = DeepAREstimator(
            prediction_length = 12,
            freq = "M",
            trainer = Trainer(epochs = 5),
        )
        self.model = deepar.train(train_set)

    def predict(self, test_set):
        # model predict
        true_values = to_pandas(list(test_set)[0])
        true_values.to_timestamp().plot(color = "k")

        prediction_input = PandasDataset([
            true_values[:-36],
            true_values[:-24],
            true_values[:-12],
        ])
        self.predictions = self.model.predict(prediction_input)

    def plot_prediction(self):
        # plotting
        for color, prediction in zip(["green", "blue", "purple"], self.predictions):
            prediction.plot(color = f"tab:{color}")
        plt.legend(["True values"], loc = "upper left", fontsize = "xx-large")
        plt.show()




# 测试代码 main 函数
def main():
    # ------------------------------
    # data
    # ------------------------------
    from gluonts.dataset.repository.datasets import get_dataset
    dataset = get_dataset("airpassengers")
    # ------------------------------
    # data split
    # ------------------------------
    from gluonts.dataset.split import split
    train_data, test_data = split(dataset, offset = -36)


if __name__ == "__main__":
    main()

