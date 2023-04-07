# -*- coding: utf-8 -*-


# ***************************************************
# * File        : MLP.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-21
# * Version     : 0.1.032122
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten() # Flatten 层将除第一维(batch_size)以外的维度”展平“
        self.dense1 = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 10)
    
    @tf.function
    def call(self, input):        # [batch_size, 28, 28, 1]
        x = self.flatten(input)    # [batch_size, 28 * 28 * 1 = 784]
        x = self.dense1(x)        # [batch_size, 100]
        x = self.dense2(x)        # [batch_size, 10]
        output = tf.nn.softmax(x) # []
        return output


model = MLP()




def main():
    pass


if __name__ == "__name__":
    pass
