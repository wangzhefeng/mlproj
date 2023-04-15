import spacy

# 1.Load the large English NLP model
nlp = spacy.load("en_core_web_lg")


# 2.Replace a token with "REDACTED"
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDACTED]"
    else:
        return token.string


# 3.Loop through all the entities in a document and check if they are names.
def scrub(text):
    doc = nlp(text)
    for ent in doc.ents:
        ent.merge()
        # ent.retokenize()
    tokens = map(replace_name_with_placeholder, doc)
    return "".join(tokens)


if __name__ == "__main__":
    s = """
    In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s 
    Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
    """
    print(scrub(s))
