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

# 4.提取所有命名实体
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")
