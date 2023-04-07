# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba.analyse import textrank
import functools


"""
训练一个关键词提取算法步骤：

   - 1.加载已有的文档数据集

   - 2.加载停用词表

   - 3.对数据集中的文档进行 **分词**

   - 4.根据停用词表，过滤干扰词

   - 5.根据数据集训练关键词提取算法

根据训练好的关键词提取算法对新文档进行关键词提取步骤：

   - 1.对新文档进行分词

   - 2.根据停用词表，过滤干扰词

   - 3.根据训练好的算法提取关键词

https://github.com/nlpinaction/code/tree/master/chapter-5
"""

project_path = os.path.abspath(".")
stopword_path = os.path.join(project_path, "src/nlp_src/utils/word_stop.txt")
corpus_path = os.path.join(project_path, "src/nlp_src/data/corpus.txt")


# 1.加载已有的文档数据集
def load_train_data():
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'

    return text


def load_data(data_path, stopword_list, pos = False):
    """
    1.加载新的文本数据
    2.对新文本进行分词
    3.对分词后的文本进行停用词过滤

    Args:
        data_path ([type]): [description]
        pos (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    doc_list = []
    for line in open(data_path, "r", encoding = "utf-8"):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, stopword_list, pos)
        doc_list.append(filter_list)

    return doc_list


# 2.加载停用词表
def get_stopword_list(stop_word_path):
    """
    停用词表存储录路径，每一行为一个词，按行读取进行加载.
        - 进行编码转换确保匹配准确率
        - 返回停用词的列表

    Args:
        stop_word_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    stopword_list = [sw.replace("\n", "") for sw in open(stop_word_path, encoding = "utf-8").readlines()]

    return stopword_list


# 3.对数据集中的文档进行 **分词**
def seg_to_list(sentence, pos = False):
    """
    分词方法，调用jieba接口

    Args:
        sentence ([type]): [description]
        pos (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    
    return seg_list


# 4.根据停用词表，过滤干扰词
def word_filter(seg_list, stopword_list, pos = False):
    """
    根据停用词表，过滤干扰词

    Args:
        seg_list ([type]): [description]
        stopword_list ([type]): [description]
        pos (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = "n" # 不进行词性过滤，则将词性都标注为 'n'，表示全部保留
        else:
            word = seg.word
            flag = seg.flag
        
        if not flag.startswith("n"):
            continue
        
        # 过滤停用词表中的词，以及长度 < 2 的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)
    
    return filter_list


# IDF 值统计方法
def train_idf(doc_list):
    idf_dict = {}
    
    # 总文档数
    tt_count = len(doc_list)

    # 每个词出现的文档树
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] = idf_dict.get(word, 0.0) + 1.0
    
    # 按公式转换为 IDF 值，分母加1进行平滑处理
    for key, value in idf_dict.items():
        idf_dict[key] = math.log(tt_count / (1.0 + value))
    
    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认IDF值
    default_idf = math.log(tt_count / 1.0)

    return idf_dict, default_idf


# 排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# ---------------------------------
# 关键字提取模型
# ---------------------------------
# TF-IDF 类
class TfIdfModel(object):

    def __init__(self, idf_dict, default_idf, word_list, keyword_num):
        """[summary]

        Args:
            idf_dict ([type]): 训练好的idf字典
            default_idf ([type]): 默认idf值
            word_list ([type]): 处理后的待提取文本
            keyword_num ([type]): 关键词数量
        """
        self.idf_dict = idf_dict
        self.default_idf = default_idf
        self.word_list = word_list
        self.keyword_num = keyword_num
        self.tf_dict = self.get_tf_dict()
    
    def get_tf_dict(self):
        """
        统计 TF 值

        Returns:
            [type]: [description]
        """
        tf_dict = {}
        for word in self.word_list:
            tf_dict[word] = tf_dict.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for key, value in tf_dict.items():
            tf_dict[key] = float(value) / tt_count
        
        return tf_dict

    def get_tfidf(self):
        """
        按公式计算 TF-IDF
        """
        tfidf_dict  = {}
        for word in self.word_list:
            idf = self.idf_dict.get(word, self.default_idf)
            tf = self.tf_dict.get(word, 0)
            tfidf = tf * idf
            tfidf_dict[word] = tfidf
        
        tfidf_dict.items()
        # 根据 TF-IDF 排序，取排名前 keyword_num 的词作为关键字
        for key, value in sorted(tfidf_dict.items(), key = functools.cmp_to_key(cmp), reverse = True)[:self.keyword_num]:
            print(key + "/ ", end = "")
        print()


# LSI, LDA 主题模型
class TopicModel(object):
    
    def __init__(self, doc_list, keyword_num, model = "LSI", num_topics = 4):
        """
        Args:
            doc_list ([type]): 处理后的数据集
            keyword_num ([type]): 关键词数量
            model (str, optional): 具体模型（LSI、LDA）. Defaults to "LSI".
            num_topics (int, optional): 主题数量. Defaults to 4.
        """
        self.dictionary = corpora.Dictionary(doc_list)              # 使用gensim的接口corpora将文本转为向量化表示，先构建词空间
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list] # 使用 BOW 模型向量化
        self.tfidf_model = models.TfidfModel(corpus)                # 对每个词，根据 TF-IDF 进行加权，得到加权后的向量表示
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        if model == "LSI":
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        word_dict = self.word_dictionary(doc_list)
        self.wordtopic_dict = self.get_wordtopic(word_dict)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word = self.dictionary, num_topics = self.num_topics)
        
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word = self.dictionary, num_topics = self.num_topics)

        return lda

    def get_wordtopic(self, word_dict):
        wordtopic_dict = {}
        for word in word_dict:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dict[word] = wordtopic
        
        return wordtopic_dict

    def get_simword(self, word_list):
        """
        计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词

        Args:
            word_list ([type]): [description]
        """
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        def calsim(l1, l2):
            """
            余弦相似度计算

            Args:
                l1 ([type]): [description]
                l2 ([type]): [description]
            """
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0

            return sim
        
        # 计算输入文本和每个词的主题分布相似度
        sim_dict = {}
        for key, value in self.wordtopic_dict.items():
            if key not in word_list:
                continue
            sim = calsim(value, senttopic)
            sim_dict[key] = sim
        
        for key, value in sorted(sim_dict.items(), key = functools.cmp_to_key(cmp), reverse = True)[:self.keyword_num]:
            print(key + "/ ", end = '')
        print()


    def word_dictionary(self, doc_list):
        """
        词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法

        Args:
            doc_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        
        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        
        return vec_list



# TF-IDF 关键字提取
def tfidf_extract(doc_list, word_list, pos = False, keyword_num = 10):
    idf_dict, default_idf = train_idf(doc_list)
    tf_idf_model = TfIdfModel(idf_dict, default_idf, word_list, keyword_num)
    tf_idf_model.get_tfidf()


# TextRank 关键字提取
def textrank_extract(text, pos = False, keyword_num = 10):
    keywords = textrank(text, keyword_num)
    for keyword in keywords:
        print(keyword + "/", end = "")
    print()


# LSI, LDA 关键字提取
def topic_extract(doc_list, word_list, model, pos = False, keyword_num = 10):
    topic_model = TopicModel(doc_list, keyword_num, model = model)
    topic_model.get_simword(word_list)











if __name__ == "__main__":
    text = load_train_data()
    pos = True
    # 停用词
    stopword_list = get_stopword_list(stop_word_path = stopword_path)
    # 分词
    seg_list = seg_to_list(text, pos)
    # 根据停用词去除分词后的语料干扰词
    filter_list = word_filter(seg_list, stopword_list, pos)
    doc_list = load_data(corpus_path, stopword_list, pos)

    print("TF-IDF 模型结果：")
    tfidf_extract(doc_list, filter_list)

    print("TextRank 模型结果：")
    textrank_extract(text)

    print("LSI 模型结果：")
    topic_extract(doc_list, filter_list, "LSI", pos)

    print("LDA 模型结果：")
    topic_extract(doc_list, filter_list, "LDA", pos)
