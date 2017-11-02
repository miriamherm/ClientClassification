# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:14:08 2017

@author: Miriam
"""

import os
import sys
import numpy as np
import pickle
import random
from functools import reduce
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, model_from_json
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix



import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

#first argument base directory for project
#second argument model name folder, with subfolders with models for each of the data_types
#third argument experiment #
BASE_DIR = os.getcwd()
TEST_DATA_DIR= os.path.join(BASE_DIR, "test_data")
types=""#sys.argv[1]
#exper=sys.argv[2]
MODEL_DIR= os.path.join(BASE_DIR, "Models/glove.6B.100d.txt/4/")

MAX_SEQUENCE_LENGTH = 10

data_types = ["names","companies", "address", "products"] #


def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**.5) + 1) if n % i == 0)))
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def read_data(TEXT_DATA_DIR, name):

    texts = []  # list of text samples
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(fpath):
        return
    if sys.version_info < (3,):
        f = open(fpath)
    else:
        f = open(fpath, encoding='latin-1')
    for t in f.readlines():
        process(texts, t)
    f.close()
    return texts

def process(texts, t):
    if not is_english(t):
        return
    if (t.strip() == ''):
        return
    num_tokens = len(t.strip().split(sep=' '))
    if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
        texts.append(t)

def load_model(model_dir, data_type):

    #load json model
    json_file = open(model_dir+data_type+'_nermodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir+data_type+"_nermodel.h5")

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #load tokenizer
    tokenizer=None
    tokenizer_file = open(model_dir+data_type+'_tokenizer.pkl', 'rb')

    tokenizer = pickle.load(tokenizer_file)

    return loaded_model, tokenizer
def model_fn(texts):
    print(types)
    loaded_model, tokenizer=load_model(MODEL_DIR+types+"/", types)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return loaded_model.predict(data)

def  main():
    preds = dict()

    for name in sorted(os.listdir(TEST_DATA_DIR)):
        #print("Loading data "+ name)
        if types in name:
           print("testing " + name + " and model " + types)
        else:
           continue 
        texts = read_data(TEST_DATA_DIR, name)
        if not texts:
            continue
        lime_texts= " ".join(texts)
        explainer = LimeTextExplainer(split_expression=" ",class_names=["not"+ types, types])
        exp = explainer.explain_instance(lime_texts, model_fn, num_features=100)
        thefile = open(BASE_DIR+'/lime/'+types+".txt", 'w')
        for item in exp.as_list():
            #print(type(item))
            thefile.write("%s\n" % (item,))
        #print(exp.as_list())
        
            
#loaded_model, tokenizer=load_model(MODEL_DIR+data_type+"/", data_type)

            #sequences = tokenizer.texts_to_sequences(texts)
            #data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            #preds[data_type] = loaded_model.predict(data)

        #for i in preds:
            #np.savetxt("output.txt", preds[i], delimiter="\t")
            #print(i, np.average(preds[i][:,0]))

if __name__ == "__main__":
    for data_type in data_types:
        types=data_type
        main()
