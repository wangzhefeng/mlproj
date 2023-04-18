#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras import layers, models, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


base_dir = "/Users/zfwang/project/deeplearning_project/imdb"
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")
train_dir = os.path.join(data_dir, "aclImdb/train")
test_dir = os.path.join(data_dir, "aclImdb/test")
glove_dir = os.path.join(data_dir, "glove.6B")


# 处理 IMDB 原始数据的标签
texts = []
labels = []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)

# 对 IMDB 原始数据的文本进行分词
maxlen = 100                # 在100个单词后阶段评论
max_words = 10000           # 只考虑数据集中前10000个最常见的单词
training_samples = 200      # 在200个样本上训练
validation_samples = 10000  # 在10000个样本上验证

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
data = pad_sequences(sequences, maxlen = maxlen)
labels = np.asarray(labels)
print("Shape of data tensor:", data.shape)
print("Shape of labels tensor:", labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
train_data = data[:training_samples]
train_labels = labels[:training_samples]
validation_data = data[training_samples:training_samples + validation_samples]
validation_labels = labels[training_samples:training_samples + validation_samples]

# 解析GloVe 词嵌入文件
embeddings_index = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = "float32")
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))

# 准备GloVe 词嵌入矩阵
embedding_dim = 100
embedding_matrix = np.zeros(shape = (max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 模型定义
model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length = maxlen)) # (samples, maxlen, 8)
model.add(layers.Flatten())                                                  # (samples, maxlen * 8)
model.add(layers.Dense(32, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))                           # 分类器
model.summary()

# 将预训练的词嵌入加载到 Embedding 层中
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# 模型训练和评估
model.compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = ["acc"]
)
history = model.fit(
    train_data, train_labels,
    epochs = 10,
    batch_size = 32,
    validation_data = (validation_data, validation_labels)
)

# 模型保存
model_path = os.path.join(model_dir, "pred_trained_glove_model.h5")
model.save_weights(model_path)

# 模型结果可视化
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label = "Training acc")
plt.plot(epochs, val_acc, "b", label = "Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label = "Train loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# 对测试数据集进行分词
texts = []
labels = []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
test_data = pad_sequences(sequences, maxlen = maxlen)
test_labels = np.asarray(labels)
print("Shape of data tensor:", test_data.shape)
print("Shape of labels tensor:", test_labels.shape)

model.load_weights(model_path)
model.evaluate(test_data, test_labels)