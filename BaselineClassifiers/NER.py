import nltk
import pandas as pd
import argparse
import sys

def extract_entity_names(t, entity_names):

    if hasattr(t, 'label'):
        if t.label() == 'PERSON' or t.label() == "GPE" or t.label() == "ORGANIZATION":
            entity_names.add(t.label())
        else:
            for child in t:
                extract_entity_names(child, entity_names)

def detect_entities(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    ents_2 = set()
    extract_entity_names(entities, ents_2)
    line=""
    print(sentence, ents_2, len(ents_2)==1)
    if len(ents_2)==1:
        line=sentence+"\t " + next(iter(ents_2))
    elif len(ents_2)==0:
        line=sentence +"\t" + " O"
    else:
        line=sentence+"\t"+" Mixed"
    with open(BASE_DIR+"\\test_companies_NLTK_Results.txt", "a", encoding="utf-8") as f1:
        f1.writelines(line + "\n")
    f1.close()
    return ents_2


def detect_column_entities(column):
    counts = {}
    for c in column:
        try:
            ents = detect_entities(c)
            print(ents)
        except:
            pass
        for e in ents:
            if e in counts:
                counts[e] += 1
            else:
                counts[e] = 0
    return counts

"""
  Temp, will disappear with a connection to the db
"""
def handleColumn(filename):
    df = pd.read_csv(filename, sep="\t", encoding='latin-1')
    for i in df.select_dtypes(include=['object']):
        print(i, detect_column_entities(df[i]))

if __name__ == '__main__':
    BASE_DIR=sys.argv[1]
    filename=BASE_DIR+"\\test_companies.txt"

    handleColumn(filename)


