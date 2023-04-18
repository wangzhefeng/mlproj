# -*- coding: utf-8 -*-
import jieba.posseg as psg


def get_part_of_speech_taging(sentence, HMM = True):
    """
    词性标注
    Params:
        HMM=False: 非 HMM 词性标注
        HMM=True: HMM 词性标注
    """
    segment_list = psg.cut(sentence, HMM)
    tagged_sentence = " ".join([f"{w}/{t}" for w, t in segment_list])
    
    return tagged_sentence


if __name__ == "__main__":
    # data
    sentence = "中文分词是文本处理不可或缺的一步!"
    tagged_sentence = get_part_of_speech_taging(sentence)
    print(tagged_sentence)
