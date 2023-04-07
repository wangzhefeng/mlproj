# -*- coding: utf-8 -*-
import jieba


"""
jieba 分词示例
"""

# 启动paddle模式
# jieba.enable_paddle()
strs = ["我来到北京清华大学", "乒乓球拍卖完了", "中国科学技术大学"]


# -------------------
# 使用paddle模式
# -------------------
# for str in strs:
#     seg_list = jieba.cut(str, use_paddle = True)
#     print("Paddle Mode: " + "/".join(list(seg_list)))


# -------------------
# 全模式
# -------------------
seg_list = jieba.cut(
    "我来到北京清华大学", 
    cut_all = True,
    HMM = True,
    use_paddle = False
)
seg_list_2 = jieba.lcut(
    "我来到北京清华大学", 
    cut_all = True,
    HMM = True,
    use_paddle = False
)
print(f"Full Mode: {'/ '.join(seg_list)}")
print(f"Full Mode: {seg_list_2}")


# -------------------
# 精确模式
# -------------------
seg_list = jieba.cut("我来到北京清华大学", cut_all = False, HMM = True, use_paddle = False)
seg_list_2 = jieba.lcut("我来到北京清华大学", cut_all = False, HMM = True, use_paddle = False)
print("Default Mode: " + "/ ".join(seg_list))
print(f"Default Mode: {seg_list_2}")


# -------------------
# 默认是精确模式
# -------------------
seg_list = jieba.cut("他来到了网易杭研大厦")
seg_list_2 = jieba.lcut("他来到了网易杭研大厦")
print(", ".join(seg_list))
print(seg_list_2)


# -------------------
# 搜索引擎模式
# -------------------
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM = False)
seg_list_2 = jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM = False)
seg_list_HMM = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM = True)
print(", ".join(seg_list))
print(seg_list_2)
print(", ".join(seg_list_HMM))
