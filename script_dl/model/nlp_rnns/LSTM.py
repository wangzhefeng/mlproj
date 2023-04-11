# -*- coding: utf-8 -*-


# ***************************************************
# * File        : lstm.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032307
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from cv_data.MNIST import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 100
learning_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
train_dataset, test_dataset = get_dataset(
    train_transform = transforms.ToTensor(),
    test_transform = transforms.ToTensor(), 
)
train_loader, test_loader = get_dataloader(
    train_dataset,
    test_dataset,
    batch_size = batch_size,
)
print(train_dataset.data.size())
print(train_dataset.data.size())
print(test_dataset.data.size())
print(test_dataset.data.size())

# ------------------------------
# model
# ------------------------------
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim) -> None:
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # init hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # init cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 28 time steps
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # index hidden state of last time step
        # out[:, -1, :].size = (100, 100)
        # out.size = (100, 10)
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# model: LSTM 1 hidden layer
# ------------------------------
# model parameters
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10
# model
lstm_one_hidden = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
lstm_one_hidden.to(device)
# loss
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(lstm_one_hidden.parameters(), lr = learning_rate)

# ------------------------------
# model: LSTM 2 hidden layer
# ------------------------------
# model parameters
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10
# model
lstm_two_hidden = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
lstm_two_hidden.to(device)
# loss
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(lstm_two_hidden.parameters(), lr = learning_rate)

# ------------------------------
# model: LSTM 3 hidden layer
# ------------------------------
# model parameters
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
input_dim = 28
hidden_dim = 100
layer_dim = 3
output_dim = 10
# model
lstm_three_hidden = LSTM(input_dim, hidden_dim, layer_dim, output_dim)
lstm_three_hidden.to(device)
# loss
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(lstm_three_hidden.parameters(), lr = learning_rate)

# ------------------------------
# model training
# ------------------------------
# num of step to unroll
seq_dim = 28
# training loop
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images to torch.tensor with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)
        # forward
        outputs = lstm_one_hidden(images)
        loss = loss_fn(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        iter += 1
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                # resize images
                images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
                labels = labels.to(device)
                # forward
                outputs = lstm_one_hidden(images)
                # predict
                _, predicted = torch.max(outputs.data, 1)
                # accuracy
                total += labels.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print(f"Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}")








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
