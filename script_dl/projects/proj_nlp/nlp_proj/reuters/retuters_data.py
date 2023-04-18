#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import models, layers
from keras.datasets import reuters
from keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
print(train_data.shape)
print(len(train_data))
print(train_data[0])
print(train_labels[0])

print(test_data.shape)
print(len(test_labels))
print(test_labels)

# 将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for key, value in word_index.items()]
)
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_newswire)