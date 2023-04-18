# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-22
# * Version     : 0.1.032215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import random
from typing import List

import numpy as np
from data import train_data, test_data

from rnn import RNN

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 词典
# ------------------------------
# 构建词典(vocabulary)
vocab = list(set([w for text in train_data.keys() for w in text.split(" ")]))
vocab_size = len(vocab)
print(f"{vocab_size} unique words found.")
# 给词典创建索引
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# ------------------------------
# input one-hot pres
# ------------------------------
def createInputs(text: str) -> List:
    """
    Returns an array of one-hot vectors representing the words
    in the input text string.

    Args:
        text (str): string

    Returns:
        List: Each one-hot vector has shape (vocab_size, 1)
    """
    inputs = []
    for w in text.split(" "):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    
    return inputs

# ------------------------------
# RNN 前向传播
# ------------------------------
def softmax(xs):
    """
    Softmax Function
    """
    return np.exp(xs) / sum(np.exp(xs))


rnn = RNN(vocab_size, 2)


def processData(data, backprop = True):
    """
    Returns the RNN's loss and accuracy for the given data.
        - data is a dictionary mapping text to True or False.
        - backprop determines if the backward phase should be run.

    Args:
        data (_type_):  a dictionary mapping text to True or False.
        backprop (bool, optional): backprop determines if the backward phase should be run.. Defaults to True.

    Returns:
        _type_: _description_
    """
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)
        # Forward
        out, _ = rnn.forward(inputs)
        probs = softmax(out)
        # Calculate loss / accuracy
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)
        if backprop:
            # Build dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            # Backward
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


# Training loop
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
