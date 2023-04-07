# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LSTMTextGenerator.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-03
# * Version     : 0.1.040319
# * Description : description
# * Link        : https://coderzcolumn.com/tutorials/artificial-intelligence/text-generation-using-pytorch-lstm-networks-and-character-embeddings#2
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMTextGenerator(nn.Module):

    def __init__(self, vocab, embed_len, hidden_dim, n_layers) -> None:
        super(LSTMTextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.word_embedding = nn.Embedding(
            num_embeddings = len(vocab),
            embedding_dim = embed_len,
        )
        self.lstm = nn.LSTM(
            input_size = embed_len, 
            hidden_size = hidden_dim,
            num_layers = n_layers,
            batch_first = True
        )
        self.linear = nn.Linear(hidden_dim, len(vocab))
    
    def forward(self, x):
        embedding = self.word_embedding(x)
        hidden = torch.randn(
            self.n_layers, 
            len(x),
            self.hidden_dim,
        ).to(device)
        carry = torch.randn(
            self.n_layer, 
            len(x), 
            self.hidden_dim
        ).to(device)
        output, (hidden, carry) = self.lstm(embedding, (hidden, carry))
        out = self.linear(output[:, -1])
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
