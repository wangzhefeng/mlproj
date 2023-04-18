# -*- coding: utf-8 -*-


import os
from os.path import isfile, join
import re
import numpy as np
from random import randint
# import tensorflow as tf
import config

num_dimensions = 300  # Dimensions for each word vector
batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 100
lr = 0.001


def load_wordsList():
    """
    载入词典，该词典包含 400000 个词汇
    """
    wordsList = np.load(os.path.join(config.root_dir, 'wordsList.npy'))
    print("-" * 20)
    print('载入word列表...')
    print("-" * 20)
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]

    return wordsList


def load_wordVectors():
    """
    载入已经训练好的词典向量模型，该矩阵包含了的文本向量，维度: (400000, 50)
    """
    wordVectors = np.load(os.path.join(config.root_dir, 'wordVectors.npy'))
    print("-" * 20)
    print('载入文本向量...')
    print("-" * 20)

    return wordVectors


def load_idsMatrix():
    """
    载入索引矩阵
    """
    ids = np.load(os.path.join(config.root_dir, 'idsMatrix.npy'))

    return ids


def postive_analysis():
    """
    载入正面数据集
    """
    pos_files = [config.pos_data_dir + f for f in os.listdir(config.pos_data_dir) if isfile(join(config.pos_data_dir, f))]
    num_words = []
    for pf in pos_files:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print("-" * 20)
    print('正面评价数据加载完结...')
    print("-" * 20)
    num_files = len(num_words)
    print('正面评价数据文件总数', num_files)
    print('正面评价数据所有的词的数量', sum(num_words))
    print('正面评价数据平均文件词的长度', sum(num_words) / len(num_words))
    
    return pos_files, num_words, num_files


def negtive_analysis():
    """
    载入负面数据集
    """
    neg_files = [config.neg_data_dir + f for f in os.listdir(config.neg_data_dir) if isfile(join(config.neg_data_dir, f))]
    num_words = []
    for nf in neg_files:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print("-" * 20)
    print('负面评价数据加载完结...')
    print("-" * 20)
    num_files = len(num_words)
    print('负面评价数据文件总数', num_files)
    print('负面评价数据所有的词的数量', sum(num_words))
    print('负面评价数据平均文件词的长度', sum(num_words) / len(num_words))

    return neg_files, num_words, num_files


def data_visual(num_words):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("qt4agg")
    # 指定默认字体
    # mpl.rcParams["font.sans-serif"] = ["SimHei"]
    # mpl.rcParams["font.family"] = "sans-serif"
    # %matplotlib inline
    plt.hist(num_words, 50, facecolor = "g")
    plt.xlabel("文本长度")
    plt.ylabel("频次")
    plt.axis([0, 1200, 0, 8000])
    plt.show()


def cleanSentences(string):
    """
    
    """
    string = string.lower().replace("<br />", " ")
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    return re.sub(strip_special_chars, "", string.lower())


def get_index_matrix(num_files):
    """
    生成一个索引矩阵，25000 * 300
    """
    # 由于大部分文本长度都在 230 之内
    max_seq_num = 300
    ids = np.zeros((num_files, max_seq_num), dtype='int32')
    file_count = 0
    # pos
    for pf in pos_files:
        with open(pf, "r", encoding='utf-8') as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[file_count][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[file_count][indexCounter] = 399999  # 未知的词
                indexCounter = indexCounter + 1
                if indexCounter >= max_seq_num:
                    break
            file_count = file_count + 1
    
    # neg
    for nf in neg_files:
        with open(nf, "r",encoding='utf-8') as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[file_count][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[file_count][indexCounter] = 399999  # 未知的词语
                indexCounter = indexCounter + 1
                if indexCounter >= max_seq_num:
                    break
            file_count = file_count + 1

    # 保存索引矩阵到文件
    # np.save(os.path.join(config.root_dir, 'idsMatrix'), ids)


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


tf.reset_default_graph()
# 输出
labels = tf.placeholder(tf.float32, [batch_size, num_labels])
# 输入
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])

data = tf.Variable(tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype = tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

# LSTM网络
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()
with tf.Session() as sess:
    if os.path.exists("models") and os.path.exists("models/checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

    iterations = 100
    for step in range(iterations):
        next_batch, next_batch_labels = get_test_batch()
        if step % 20 == 0:
            print("step:", step, " 正确率:", (sess.run(
                accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    print("Model saved in path: %s" % save_path)



if __name__ == "__main__":
    # 词典
    wordsList = load_wordsList()
    print("词典中词汇数量:", len(wordsList))
    home_index = wordsList.index("home")
    print("'home' 单词在词典中的索引:", home_index)
    
    # 词典向量模型矩阵
    wordVectors = load_wordVectors()
    print("词典向量模型矩阵:", wordVectors.shape)
    print("'home' 在词典向量模型矩阵中的向量表示:", wordVectors[home_index])

    # 正面、负面文本数据
    pos_files, pos_num_words, pos_num_files = postive_analysis()
    neg_files, neg_num_words, neg_num_files = negtive_analysis()

    # 文本数据可视化
    # num_total_words = pos_num_words + neg_num_words
    # data_visual(num_total_words)


    # 生成索引矩阵
    num_total_files = pos_num_files + neg_num_files
    get_index_matrix(num_total_files)
