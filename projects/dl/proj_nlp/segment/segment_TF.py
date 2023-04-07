# -*- coding: utf-8 -*-


import os
import config
import glob
import random
import jieba


"""
提取高频词

    高频词：
        - 高频词一般指文档中出现频率较高且非无用的词语, 其一定程度上代表了文档的焦点所在。
        - 针对单篇文档, 可以作为一种关键词来看
        - 对于如新闻这样的多篇文档, 可以将其作为热词, 发现舆论焦点
    高频词提取：
        - 高频词提取其实就是 NLP 中的 TF(Term Frequency)策略
        - 高频词提取有以下干扰项：
            - 标点符号：一般标点符号无任何价值, 需要去除
            - 停用词：诸如“的”、“是”、“了”等常用词无任何意义, 也需要剔除
    示例：
        - 采用 Jieba 分词, 针对搜狗实验室的新闻数据, 进行高频词提取
        - 数据 news：
            - 包括9个目录, 目录下均为 txt 文件, 分别代表不同领域的新闻, 
            - 该数据本质上是一个分类语料, 这里只挑选其中一个类别, 统计该类的高频词
"""


def get_content(path):
    """
    读取数据
    """
    with open(path, "r", encoding = "gbk", errors = "ignore") as f:
        content = ""
        for l in f:
            l = l.strip()
            content += l
        return content


def get_TF(words, topK = 50):
    """
    获取 topK 高频词汇
    """
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    term_frequeny = sorted(tf_dic.items(), key = lambda x: x[1], reverse = True)[:topK]
    return term_frequeny


def get_stop_words(path = os.path.join(config.utils_dir, "stop_words.utf8")):
    with open(path) as f:
        return [l.strip() for l in f]


def main(texts = "news/C000013/*.txt"):
    """
    获取语料、分词
    """
    files = glob.glob(os.path.join(config.data_dir, texts))
    corpus = [get_content(x) for x in files]
    for i in range(len(corpus)):
        corpus_i = corpus[i]
        split_words = [x for x in jieba.cut(corpus_i) if x not in get_stop_words()]
        print("样本之一：" + corpus_i, "\n")
        print("样本分词效果：" + ", ".join(split_words), "\n")
        print("样本的 topK(10)词：" + str(get_TF(split_words)), "\n")






if __name__ == "__main__":
    stop_words = get_stop_words()
    print(stop_words)
