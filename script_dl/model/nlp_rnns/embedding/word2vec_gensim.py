# -*- coding: utf-8 -*-


# ***************************************************
# * File        : word2vec_gensim.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032722
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import codecs
import jieba

import opencc
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 项目目录
project_path = os.path.abspath(".")
data_path = "/Users/zfwang/project/machinelearning/deeplearning/data/NLP_data/word2vec_data"

# 维基百科中文语料
corpus_path = os.path.join(data_path, "zhwiki-latest-pages-articles.xml.bz2")

# Gensim 数据提取
extracted_corpus_path = os.path.join(data_path, "wiki-zh-article.txt")

# 繁体中文转换为简体中文
extracted_zhs_corpus_path = os.path.join(data_path, "wiki-zh-article-zhs.txt")

# 分词
cuted_word_path = os.path.join(data_path, "wiki-zh-words.txt")

# 模型
model_output_1 = os.path.join(project_path, "src/nlp_src/models/wiki-zh-model")
model_output_2 = os.path.join(project_path, "src/nlp_src/models/wiki-zh-vector")

# ----------------------------------------
# 1.数据预处理
# ----------------------------------------
# TODO 判断目录中是否存在相关数据文件
space = " "
with open(extracted_corpus_path, "w", encoding = "utf8") as f:
    wiki = WikiCorpus(corpus_path, lemmatize = False, dictionary = {})
    for text in wiki.get_texts():
        print(text)
        f.write(space.join(text) + "\n")
print("Finished Saved.")
# ----------------------------------------
# 2.繁体字处理
# ----------------------------------------
t2s_converter = opencc.OpenCC("t2s.json")
with open(extracted_corpus_path, "r", encoding = "utf8") as f1:
    with open(extracted_zhs_corpus_path, "w", encoding = "utf8") as f2:
        extracted_zhs_corpus = t2s_converter.convert(f1)
        f2.write(extracted_zhs_corpus)
print("Finished Converter.")
# ----------------------------------------
# 3.分词
# ----------------------------------------
descsFile = codecs.open(extracted_zhs_corpus_path, "rb", encoding = "utf-8")
i = 0
with open(cuted_word_path, "w", encoding = "utf-8") as f:
    for line in descsFile:
        i += 1
        if i % 10000 == 0:
            print(i)
        line = line.strip()
        words = jieba.cut(line)
        for word in words:
            f.write(word + " ")
        f.write("\n")
# ----------------------------------------
# 4.运行 word2vec 训练模型
# ----------------------------------------
model = Word2Vec(
    LineSentence(cuted_word_path), 
    size = 400, 
    window = 5, 
    min_count = 5, 
    workers = multiprocessing.cpu_count()
)
model.save(model_output_1)
model.save_word2vec_format(model_output_2, binary = False)
# ----------------------------------------
# 5.模型测试
# ----------------------------------------
model = Word2Vec(model_output_1)
# model = Word2Vec.load_word2vec_format(model_output_2, binary = False)
res = model.most_similar("时间")
print(res)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
