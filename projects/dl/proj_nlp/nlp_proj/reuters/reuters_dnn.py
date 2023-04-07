#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models, layers, preprocessing
from keras.utils.np_utils import to_categorical


# ----------------------
# data
# ----------------------
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

# ----------------------
# data preprocessing[one-hot encode]
# ----------------------
# ---------------
# 将标签向量化
# ---------------
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        
        return results

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)

# ---------------
# 将标签向量化
# ---------------
# method 1
# def to_one_hot(labels, dimension = 46):
#     results = np.zeros(len(labels), dimension)
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
    
#     return results

# train_labels = to_one_hot(train_labels)
# test_labels = to_one_hot(test_labels)

# method 2
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# method3 
train_labels_sparse = np.array(train_labels)
test_labels_sparse = np.array(test_labels)



# ---------------
# dataset split into trainging dataset and validation dataset
# ---------------
validation_data = train_data[:1000]
validation_labels = train_labels[:1000]
partial_train_data = train_data[1000:]
partial_train_labels = train_labels[1000:]

# ----------------------
# model
# ----------------------
model = models.Sequential()
model.add(layers.Dense(64, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(46, activation = "softmax"))

# ----------------------
# model compile
# ----------------------
model.compile(optimizer = "rmsprop",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# model.compile(optimizer = "rmsprop",
#               loss = "sparse_categorical_crossentropy",
#               metrics = ["acc"])
# ----------------------
# model training on trainging dataset and validation datasets
# ----------------------
history = model.fit(partial_train_data, 
                    partial_train_labels,
                    epochs = 20, 
                    batch_size = 512,
                    validation_data = (validation_data, validation_labels))

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

# model_visualer(history)

# ----------------------
# model evaluate on testing dataset
# ----------------------
model = models.Sequential()
model.add(layers.Dense(64, activation = "relu", input_shape = (10000,)))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(46, activation = "softmax"))

model.compile(optimizer = "rmsprop",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(train_data,train_labels, 
          epochs = 9,
          batch_size = 512)
results = model.evaluate(test_data, test_labels)
print(results)

# test dataset 完全随机的结果
import copy 
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array)))



# ----------------------
# testing dataset predict result
# ----------------------
predictions = model.predict(test_data)
print(np.argmax(predictions[0]))
