#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.datasets import imdb
from keras import layers, models, preprocessing
import matplotlib.pyplot as plt


max_features = 10000
maxlen = 20
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = max_features)
train_data = preprocessing.sequence.pad_sequences(train_data, maxlen = maxlen) # (samples, maxlen)
test_data = preprocessing.sequence.pad_sequences(test_data, maxlen = maxlen)   # (samples, maxlen)


model = models.Sequential()
model.add(layers.Embedding(10000, 8, input_length = maxlen))# (samples, maxlen, 8)
model.add(layers.Flatten())                                 # (samples, maxlen * 8)
model.add(layers.Dense(1, activation = "sigmoid"))          # 分类器
model.compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = ["acc"]
)
model.summary()
history = model.fit(
    train_data, train_labels,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2
)

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



if __name__ == "__main__":
    pass