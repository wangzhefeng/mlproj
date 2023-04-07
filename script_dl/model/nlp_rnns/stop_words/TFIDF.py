import numpy as np
import pandas as pd
import math
import jieba
import jieba.analyse
import re



def word_segment(file_path_before, file_path_after, stop_word_file, dic_file):
    """
    中文分词
    """
    # 读取待编码的文件
    with open(file_path_before, encoding = "utf-8") as f:
        docs = f.readlines()
    
    # 加载停用词
    stopwords=[]
    for word in open(stop_word_file, 'r'): # 这里加载停用词的路径
        stopwords.append(word.strip("\n"))

    words_list = []
    for i,text in enumerate(docs):
        words = []
        final_word = []
        p = re.compile(r"\n|：|；|,|、|（|）|\.|。|，|/|(\|)", re.S)
        text = p.sub('', text)
        jieba.load_userdict(dic_file)
        word = jieba.cut(text)
        words += word
        for i,word in enumerate(words):
            if word not in stopwords:
                final_word.append(word)
        words_list.append(final_word)
    
    for i in range(0,len(words_list)):
        with open(file_path_after, 'a') as f1:
            f1.write(str(words_list[i]))
            f1.write("\n")

    return words_list



def doc2tfidf_matrix(file_path_before, words_list):
    """
    生成TD-IDF矩阵
    """
    # 找出分词后不重复的词语，作为词袋
    words = []
    for i,word in enumerate(words_list):
        words += word
    vocab = sorted(set(words),key=words.index)
    # print(vocab)
    
    # 建立一个M行V列的全0矩阵，M是文档样本数，这里是行数，V为不重复词语数，即编码维度
    V = len(vocab)
    M = len(words_list)
    onehot = np.zeros((M,V)) # 二维矩阵要使用双括号
    tf = np.zeros((M,V))
    
    for i,doc in enumerate(words_list):
        for word in doc:
            if word in vocab:
                pos = vocab.index(word)
                # print(pos,word)
                onehot[i][pos] = 1
                tf[i][pos] += 1 # tf,统计某词语在一条样本中出现的次数
            else:
                print(word)
    
    row_sum = tf.sum(axis=1) # 行相加，得到每个样本出现的词语数
    # 计算TF(t,d)
    tf = tf/row_sum[:, np.newaxis] #分母表示各样本出现的词语数，tf为单词在样本中出现的次数，[:,np.newaxis]作用类似于行列转置
    # 计算DF(t,D)，IDF
    df = onehot.sum(axis=0) # 列相加，表示有多少样本包含词袋某词
    idf = list(map(lambda x:math.log10((M+1)/(x+1)),df))
    # 计算TFIDF
    tfidf = tf*np.array(idf)
    tfidf = pd.DataFrame(tfidf,columns=vocab)
    return tfidf


import numpy as np
import pandas as pd
import math
import jieba

def doc2tfidf_matrix():
    # 读取待编码的文件
    file_path=input("请输入待编码文件路径及文件名：")
    with open(file_path,encoding="utf-8") as f:
        docs=f.readlines()
    
    # 将文件每行分词，分词后的词语放入words中
    words=[]
    for i in range(len(docs)):
        docs[i]=jieba.lcut(docs[i].strip("\n"))
        words+=docs[i]
    
    # 找出分词后不重复的词语，作为词袋
    vocab=sorted(set(words),key=words.index)
    
    # 建立一个M行V列的全0矩阵，M问文档样本数，这里是行数，V为不重复词语数，即编码维度
    V=len(vocab)
    M=len(docs)
    onehot=np.zeros((M,V)) # 二维矩阵要使用双括号
    tf=np.zeros((M,V))
    
    for i,doc in enumerate(docs):
        for word in doc:
            if word in vocab:
                pos=vocab.index(word)
                onehot[i][pos]=1
                tf[i][pos]+=1 # tf,统计某词语在一条样本中出现的次数

    row_sum=tf.sum(axis=1) # 行相加，得到每个样本出现的词语数
    # 计算TF(t,d)
    tf=tf/row_sum[:,np.newaxis] #分母表示各样本出现的词语数，tf为单词在样本中出现的次数，[:,np.newaxis]作用类似于行列转置
    # 计算DF(t,D)，IDF
    df=onehot.sum(axis=0) # 列相加，表示有多少样本包含词袋某词
    idf=list(map(lambda x:math.log10((M+1)/(x+1)),df))
    
    # 计算TFIDF
    tfidf=tf*np.array(idf)
    tfidf=pd.DataFrame(tfidf,columns=vocab)
    return tfidf



file_path_before = "C:/Users/asus/Desktop/goodat.txt"
file_path_after = "C:/Users/asus/Desktop/word segment.txt"
stop_word_file = "C:/Users/asus/Desktop/stop_word.txt"
dic_file = "C:/Users/asus/Desktop/dic.txt"

words_list = word_segment(file_path_before, file_path_after,stop_word_file,dic_file)
tfidf = doc2tfidf_matrix(file_path_after, words_list)
for i in range(0,50):
    goodat = ""
    row = tfidf.iloc[i].sort_values(ascending=False)
    for i in range(0,10):
        goodat = goodat + row.index[i] + "/"
    print(goodat)
