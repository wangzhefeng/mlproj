import spacy
import textacy.extract


# 1.载入 En NLP 模型
nlp = spacy.load("en_core_web_lg")

# 2.文本数据
text = """London is the capital and most populous city of England and 
    the United Kingdom.  Standing on the River Thames in the south east 
    of the island of Great Britain, London has been a major settlement 
    for two millennia. It was founded by the Romans, who named it Londinium.
    """

# 3.解析文本数据
doc = nlp(text)

# 4.提取出现过的名词(noun)块
noun_chunks = textacy.extract.noun_chunks(doc, min_freq = 3)

# 5.将名词块转换为小写字母
noun_chunks = map(str, noun_chunks)
noun_chunks = map(str.lower, noun_chunks)
# print(noun_chunks)

# 6.打印结果
for noun_chunk in set(noun_chunks):
    # if len(noun_chunk.split(" ")) > 1:
        # print(noun_chunk)
    print(noun_chunk)
