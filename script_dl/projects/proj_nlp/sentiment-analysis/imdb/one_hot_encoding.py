#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import string
from keras.preprocessing.text import Tokenizer


# --------------------------
# 单词级的 one-hot encoding 
# --------------------------
samples = ["The cat sat on the mat.", "The dog ate my homework."]
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
print(token_index)

max_length = 10
results = np.zeros(shape = (len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.0
print(results)

# --------------------------
# 字符级的 one-hot encoding 
# --------------------------
samples = ["The cat sat on the mat.", "The dog ate my homework."]
characters = string.printable # 所有可打印的 ASCII 字符
token_index = dict(zip(range(1, len(characters) + 1), characters))
print(token_index)

max_length = 50
results = np.zeros(shape = (len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.0
print(results)


# --------------------------
# 用 Keras 上限单词级的 ont-hot 编码
# --------------------------
samples = ["The cat sat on the mat.", "The dog ate my homework."]
tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode = "binary")
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
print(word_index)
print(one_hot_results)