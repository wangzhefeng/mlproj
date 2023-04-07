# -*- coding: utf-8 -*-


# ***************************************************
# * File        : imgview.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-04
# * Version     : 0.1.040417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import matplotlib.pyplot as plt
import torch


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# %matplolib inline
# %config InlineBackend.figure_format = "svg"


def rgb_img_view(dataset):
    plt.figure(figsize = (8, 8))
    for i in range(9):
        # data
        img, label = dataset[i]
        img = torch.squeeze(img).numpy()
        # plot
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"label = {label}")
    plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
