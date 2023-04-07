# -*- coding: utf-8 -*-


# ***************************************************
# * File        : rnn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-22
# * Version     : 0.1.032214
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from cv_data.MNIST import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 64
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
train_dataset, test_dataset = get_dataset(
    train_transform = transforms.ToTensor(),
    test_transform = transforms.ToTensor(), 
)
print(train_dataset.data.size())
print(train_dataset.data.size())
print(test_dataset.data.size())
print(test_dataset.data.size())

train_loader, test_loader = get_dataloader(
    train_dataset,
    test_dataset,
    batch_size = batch_size,
)

# ------------------------------
# model
# ------------------------------
class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, nonlinearity) -> None:
        super(RNN, self).__init__()
        # hidden dim
        self.hidden_dim = hidden_dim
        # num of hidden layers
        self.layer_dim = layer_dim
        # rnn
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first = True,
            nonlinearity = nonlinearity,
        )
        # readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # h0
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # detech hidden state
        out, hn = self.rnn(x, h0.detach())
        # index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# RNN with 1 hidden layer
# ------------------------------
# model parameters
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10
# model
rnn_one_hidden = RNN(input_dim, hidden_dim, layer_dim, output_dim, nonlinearity = "relu")
rnn_one_hidden.to(device)
print(rnn_one_hidden)

# ------------------------------
# RNN with 2 hidden layer
# ------------------------------
# model parameters
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10
# model
rnn_two_hidden = RNN(input_dim, hidden_dim, layer_dim, output_dim, nonlinearity = "relu")
rnn_two_hidden.to(device)
print((rnn_two_hidden))

# ------------------------------
# RNN with 2 hidden layer and tanh
# ------------------------------
# model parameters
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10
# model
rnn_two_hidden_tanh = RNN(input_dim, hidden_dim, layer_dim, output_dim, nonlinearity = "tanh")
rnn_two_hidden_tanh.to(device)
print(rnn_two_hidden_tanh)

# ------------------------------
# model training
# ------------------------------
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
# num of steps to unroll
seq_dim = 28

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(rnn_one_hidden.parameters(), lr = learning_rate)

# total groups of parameters
len(list(rnn_one_hidden.parameters()))
# Input -> Hidden weight
print(list(rnn_one_hidden.parameters())[0].size())
# Hidden -> Hidden
print(list(rnn_one_hidden.parameters())[1].size())
# Input -> Hidden bias
print(list(rnn_one_hidden.parameters())[2].size())
# Hidden -> Hidden bias
print(list(rnn_one_hidden.parameters())[3].size())
# Hidden -> Output
print(list(rnn_one_hidden.parameters())[4].size())
# Hidden -> Output Bias
print(list(rnn_one_hidden.parameters())[5].size())

# model training loop
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # model train
        rnn_one_hidden.train()
        # load images as tensor with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)
        # forward
        output = rnn_one_hidden(images)
        loss = criterion(output, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # test
        iter += 1
        if iter % 500 == 0:
            # model eval
            rnn_one_hidden.eval()
            # accuracy
            correct = 0
            total = 0
            # test
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
                labels = labels.to(device)
                # forward
                outputs = rnn_one_hidden(images)
                # predict
                _, predicted = torch.max(output.data, 1)
                # accuracy
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print(f"Iteration: {iter}, Loss: {loss.item()}, Accuracy: {accuracy}")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
