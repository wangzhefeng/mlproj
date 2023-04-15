# -*- coding: utf-8 -*-


# ***************************************************
# * File        : gru.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032308
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn
from d2l import torch as d2l


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 32
num_steps = 35


# ------------------------------
# data
# ------------------------------
train_iter, vocab = d2l.load_data_time_machine(
    batch_size = batch_size, 
    num_steps = num_steps
)


# ------------------------------
# model
# ------------------------------
def get_params(vocab_size, num_hiddens, device):
    """
     初始化模型参数

    Args:
        vocab_size (_type_): _description_
        num_hiddens (_type_): 隐藏单元的数量
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_inputs = num_outputs = vocab_size

    # 高斯分布(mmena, std = 0.01)
    def normal(shape):
        return torch.randn(size = shape, device = device) * 0.01
    
    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device = device),  # 偏置项设置为 0
        )
    
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device = device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    
    return params


def init_gru_state(batch_size, num_hiddens, device):
    """
    定义隐状态的初始化函数

    Args:
        batch_size (_type_): _description_
        num_hiddens (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (torch.zeros((batch_size, num_hiddens), device = device),)
    

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim = 0), (H,)



num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
