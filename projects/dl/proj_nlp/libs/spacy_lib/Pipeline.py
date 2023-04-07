"""
Coding the NLP Pipeline in Python

- ENV prepare

    - Install spaCy
        $ pip install spacy
    - Download the large English model for spaCy
        $ pip install ~/project/machinelearning/deeplearning/NLP/spaCy_model/en_core_web_lg.2.3.1.tar.gz
    - Install textacy which will also be useful
        $ pip install textacy
"""

import spacy
import textacy.extract

"""
建立 NLP Pipeline
"""

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


# 4.分句





# 5.分词




