#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import layers, models
from keras.datasets import imdb
from keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
print(train_data.shape)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))
print(len(train_data))
print(train_labels)

print(test_data.shape)
print(len(test_labels))
print(test_labels)

# 将 train_data 的第一条评论解码为英文单词
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for key, value in word_index.items()]
)
decoded_review = ' '.join([reverse_word_index.get(i -3, "?") for i in train_data[0]])
print(decoded_review)