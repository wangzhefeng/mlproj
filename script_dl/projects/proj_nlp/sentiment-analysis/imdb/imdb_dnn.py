#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers, preprocessing
from keras import optimizers, losses, metrics
from keras.utils import to_categorical

# -------------------
# data
# -------------------
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
print(train_data.shape)
print(train_data[0])

# -------------------
# data preprocessing[one-hot encoding]
# -------------------
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    
    return results

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)
print(train_data.shape)
print(train_data[0])

train_labels = np.asarray(train_labels).astype("float32")
test_labels = np.asarray(test_labels).astype("float32")
print(train_labels.shape)
print(train_labels[0])

# -------------------
# data split into training dataset and validation dataste
# -------------------
validation_data = train_data[:10000]
validation_labels = train_labels[:10000]
partial_train_data = train_data[10000:]
partial_train_labels = train_labels[10000:]

# -------------------
# model build
# -------------------
model = models.Sequential()
model.add(layers.Dense(16, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

# -------------------
# model compile
# -------------------
# model.compile(optimizer = "rmsprop", 
#               loss = "binary_crossentropy",
#               metrics = ["accuracy"])
model.compile(optimizer = "rmsprop", 
              loss = "binary_crossentropy",
              metrics = ["acc"])
# model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
#               loss = "binary_crossentropy",
#               metrics = ["accuracy"])
# model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
#               loss = losses.binary_crossentropy,
#               metrics = [metrics.binary_accuracy])

# -------------------
# model training
# -------------------
history = model.fit(partial_train_data, partial_train_labels, 
                    epochs = 20, 
                    batch_size = 512, 
                    validation_data = (validation_data, validation_labels))
history_dict = history.history
print(history_dict.keys())

# -------------------
# 模型训练结果在训练集合验证集上的可视化
# -------------------
def model_visualer(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label = "Training acc")
    plt.plot(epochs, val_acc, "b", label = "Validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, "bo", label = "Train loss")
    plt.plot(epochs, val_loss, "b", label = "Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()

model_visualer(history)

# -------------------
# 使用全部的训练数据训练最后的模型【模型参数经过调参】, 并在测试数据上进行结果评估
# -------------------
model = models.Sequential()
model.add(layers.Dense(16, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", 
              loss = "binary_crossentropy",
              metrics = ["acc"])

model.fit(train_data, train_labels, epochs = 4, batch_size = 512)
results = model.evaluate(test_data, test_labels)
print(results)

# -------------------
# 使用训练好的网络在新数据上生产预测结果
# -------------------
test_predict = model.predict(test_data)
print(test_predict)