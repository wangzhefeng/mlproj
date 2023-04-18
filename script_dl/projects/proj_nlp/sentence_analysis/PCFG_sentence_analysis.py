# -*- coding: utf-8 -*-
import jieba
from nltk.parse import stanford
import os

root = "./jar/"
parser_path = root + "stanford-parser.jar"
model_path = root + "stanford-parser-3.8.0-models.jar"
JAVA_HOME = "/usr/lib/jvm/jdk1.8"
JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home/bin/java"
JAVA_HOME = "/usr/bin/java"
pcfg_path = "edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz"


"""
采用 Stanford Parser 演示当下最流行的基于 PCFG 的句法分析方法
"""


def get_setment(string):
    """
    分词
    """
    seg_list = jieba.cut(string, cut_all = False, HMM = True)
    seg_str = " ".join(seg_list)
    print(seg_str)

    return seg_str


def get_PCFG(seg_str):
    """
    PCFG 句法分析
    """
    # 指定 JDK 路径
    if not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = JAVA_HOME
    
    # PCFG 模型路径
    parser = stanford.StanfordParser(
        path_to_jar = parser_path,
        path_to_models_jar = model_path,
        model_path = pcfg_path
    )
    sentence = parser.raw_parse(seg_str)
    for line in sentence:
        print(line)
        line.draw()



if __name__ == "__main__":
    string1 = "他骑自行车取了菜市场"
    string2 = "我爱北京天安门"

    # 分词
    seg_str = get_setment(string1)
    get_PCFG(seg_str)