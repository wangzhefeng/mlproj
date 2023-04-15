# -*- coding: utf-8 -*-


"""
规则分词:

    1.正向最大匹配法
    2.逆向最大匹配法
    3.双向最大匹配法
"""

class MM(object):
    """
    正向最大匹配
    """
    def __init__(self):
        self.window_size = 3

    def cut(self, text):
        result = []
        index = 0
        text_length = len(text)
        dic = ["研究", "研究生", "生命", "命", "的", "起源"]
        while text_length > index:
            for size in range(self.window_size + index, index, -1):
                piece = text[index:size]
                if piece in dic:
                    index = size - 1
                    break
            index = index + 1
            result.append(piece + "----")
        print(result)


class RMM(object):
    """
    逆向最大匹配法
    """
    def __init__(self):
        self.window_size = 3
    
    def cut(self, text):
        result = []
        index = len(text)
        dic = ["研究", "研究生", "生命", "命", "的", "起源"]
        while index > 0:
            for size in range(index - self.window_size, index):
                piece = text[size:index]
                if piece in dic:
                    index = size + 1
                    break
            index = index - 1
            result.append(piece + "----")
        result.reverse()
        print(result)


class BMM(object):
    """
    双向最大匹配法
    """
    def __init__(self):
        pass

    def cut(self, text):
        pass



if __name__ == "__main__":
    text = "研究生命的起源"
    
    MM_tokenizer = MM()
    print(MM_tokenizer.cut(text))

    RMM_tokenizer = RMM()
    print(RMM_tokenizer.cut(text))

    BMM_tokenizer = BMM()
    print(BMM_tokenizer.cut(text))
