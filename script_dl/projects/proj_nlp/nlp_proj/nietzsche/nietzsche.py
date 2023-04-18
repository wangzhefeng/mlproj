# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras


base_dir = "/Users/zfwang/project/machinelearning/deeplearning"
project_dir = os.path.join(base_dir, "deeplearning/src/project_src/nietzsche")
data_dir = os.path.join(base_dir, "datasets/nietzsche_data")
models_dir = os.path.join(project_dir, "models")
images_dir = os.path.join(project_dir, "images")


# 1.准备数据
print("-" * 100)
print("Text data...")
print("-" * 100)
text_data_path = os.path.join(data_dir, "nietzsche.txt")
path = keras.utils.get_file(fname = text_data_path, origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print("Corpus length:", len(text)) # 共有600893个字符
print()
print("text head sequences:", text[:100])
print()
print("text tail sequences:", text[-100:])


# ------------------------------------------
# data preprocessing
# ------------------------------------------
print("-" * 100)
print("Text data split...")
print("-" * 100)
maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))
print("Number of next_chars:", len(next_chars))

# one-hot encoding
print("-" * 100)
print("Vectorzation...")
print("-" * 100)
chars = sorted(list(set(text)))
print("Unique characters:", len(chars))
print("chars:", chars)
char_indices = dict((char, chars.index(char)) for char in chars)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print("x", x)
print("x shape:", x.shape)
print("y:", y)
print("y shape:", y.shape)

# ------------------------------------------
# build the model
# ------------------------------------------
model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape = (maxlen, len(chars))))
model.add(keras.layers.Dense(len(chars), activation = "softmax"))
model.compile(
    loss = keras.losses.categorical_crossentropy, 
    optimizer = keras.optimizers.RMSprop(lr = 0.01)
)


def reweight_distribution(original_distribution, temperature):
    distribution = np.log(original_distribution) / temperature
    exp_distribution = np.exp(distribution)
    distribution = exp_distribution / np.sum(exp_distribution)
    return distribution

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype("float64")
    preds = reweight_distribution(preds, temperature)
    probas = np.random.multinomial(1, preds, 1)
    new_sample = np.argmax(probas)
    return new_sample


for epoch in range(1, 60):
    print("epoch", epoch)
    model.fit(x, y, batch_size = 128, epochs = 1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index:start_index + maxlen]
    print("--- Generating with seed: '" + generated_text + "'")


for temperature in [0.2, 0.5, 1.0, 1.2]:
    print("------ temperature:", temperature)
    sys.stdout.write(generated_text)

    for i in range(400):
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        
        preds = model.predict(sampled, verbose = 0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]

        generated_text += next_char
        generated_text = generated_text[1:]
        sys.stdout.write(next_char)
