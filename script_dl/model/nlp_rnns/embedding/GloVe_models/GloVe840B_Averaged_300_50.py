# -*- coding: utf-8 -*-


# ***************************************************
# * File        : GloVe.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtext
from torchtext import datasets
from torchtext.data import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import GloVe

from EmbeddingClassifier import EmbeddingClassifier


"""
GloVe paper Model
GloVe 840B(Embedding Length = 300, Tokens per Text Example = 25)
> GloVe 840B(Embedding Length = 300, Tokens per Text Example = 50)
GloVe 42B(Embedding Length = 300, Tokens per Text Example = 50)
GloVe 840B Averaged(Embedding Length = 300, Tokens per Text Example = 50)
GloVe 840B Summed(Embedding Length = 300, Tokens per Text Example = 50)
"""


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# hyper parameters
max_words = 50
embed_len = 300
batch_size = 1024
num_epochs = 25
learning_rate = 1e-3


# ------------------------------
# 分词器
# ------------------------------
tokenizer = get_tokenizer("basic_english")

# ------------------------------
# pre-train model
# ------------------------------
global_vectors = GloVe(name = "840B", dim = embed_len)

# ------------------------------
# data
# ------------------------------
# data preprocessing
def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    # 分词
    X = [tokenizer(x) for x in X]
    # 填充空字符串(pad empty string)
    X = [
        tokens + [""] * (max_words - len(tokens)) 
        if len(tokens) < max_words
        else tokens[:max_words] 
        for tokens in X
    ]
    # 将输入转换为 tensor
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        # embedding
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    
    return X_tensor.mean(dim = 1), torch.tensor(Y) - 1

# dataset
target_classes = ["World", "Sports", "Business", "Sci/Tech"]
train_dataset, test_dataset = datasets.AG_NEWS()
train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

# dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = vectorize_batch,
)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = vectorize_batch,
)

# ------------------------------
# model training
# ------------------------------
# classifier model
embed_clf = EmbeddingClassifier(max_words = max_words, embed_len = embed_len, target_classes = target_classes)

# loss
loss_fn = nn.CrossEntropyLoss()

# otpimizer
optimizer = torch.optim.Adam(embed_clf.parameters(), lr = learning_rate)

# validition
def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            # forward
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))
        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)
        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))

# train
def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs = 10):
    """
    模型训练
    """
    for i in range(1, epochs + 1):
        losses = []
        for X, Y in tqdm(train_loader):
            # forward
            Y_preds = model(X)
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 5 == 0:
            print(f"Train Loss : {torch.tensor(losses).mean()}")
            CalcValLossAndAccuracy(model, loss_fn, val_loader)


TrainModel(
    model = embed_clf,
    loss_fn = criterion,
    optimizer = optimizer,
    train_loader = train_loader,
    val_loader = test_loader,
    epochs = num_epochs,
)



def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().numpy(), F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().numpy()

Y_actual, Y_preds = MakePredictions(embed_clf, test_loader)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))

from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np

skplt.metrics.plot_confusion_matrix(
    [target_classes[i] for i in Y_actual], [target_classes[i] for i in Y_preds],
    normalize=True,
    title="Confusion Matrix",
    cmap="Purples",
    hide_zeros=True,
    figsize=(5,5)
)
plt.xticks(rotation=90);

# 测试代码 main 函数
def main():
    # 测试数据
    test_sentence = "Hello, How are you?"

    # 分词
    test_tokens = tokenizer(test_sentence)
    print(test_tokens)

    # pre-train model
    test_embeddings = global_vectors.get_vecs_by_tokens(
        test_tokens, 
        lower_case_backup = True
    )
    print(test_embeddings.shape)
    print(test_embeddings)

    # data preview
    for X, Y in train_loader:
        print(X.shape, Y.shape)
        break

    pass

if __name__ == "__main__":
    main()
