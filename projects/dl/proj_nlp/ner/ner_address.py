# -*- coding: utf-8 -*-
def tag_line(words, mark):
    chars = []
    tags = []
    temp_word = ""
    for word in words:
        word = word.strip("\t ")
        if temp_word == "":
            bracket_pos = word.find("[")
            w, h = word.split("/")
            if bracket_pos == -1:
                if len(w) == 0:
                    continue
                chars.extend(w)
                if h == "ns":
                    tags += ["S"] if len(w) == 1 else ["B"] + ["M"] * (len(w) - 2) + ["E"]
                else:
                    tags += ["O"] * len(w)
            else:
                w = w[bracket_pos + 1:]
                temp_word += w
        else:
            bracket_pos = word.find("]")
            w, h = word.split("/")


