# -*- coding: utf-8 -*-


# ***************************************************
# * File        : OneHot.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032804
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import os
import numpy as np
import pandas as pd
import jieba
import config


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def word2onthot(file_path):
    """
    文本向量化 One-Hot
    """
    # 读取待编码的文件
    with open(file_path, encoding = "utf-8") as f:
        docs = f.readlines()
    # 将文件每行分词, 分词后的词语放入 words 中
    words = []
    for i in range(len(docs)):
        docs[i] = jieba.lcut(docs[i].strip("\n"))
        words += docs[i]
    # 找出分词后不重复的词语, 作为词袋, 是后续 onehot 编码的维度, 放入 vocab 中
    vocab = sorted(set(words), key = words.index)
    # 建立一个 M 行 V 列的全 0 矩阵, M 是文档样本数, 这里是行数, V 为不重复词语数, 即编码维度
    M = len(docs)
    V = len(vocab)
    onehot = np.zeros((M, V))
    for i, doc in enumerate(docs):
        print(i, doc)
        for word in doc:
            if word in vocab:
                pos = vocab.index(word)
                onehot[i][pos] = 1
    onehot = pd.DataFrame(onehot, columns = vocab)
    onehot.to_csv(os.path.join(config.data_dir, "onehot.csv"))
    return onehot




# 测试代码 main 函数
def main():
    corpus = os.path.join(config.data_dir, "corpus.txt")
    onehot = word2onthot(corpus)
    print(onehot)

if __name__ == "__main__":
    main()
