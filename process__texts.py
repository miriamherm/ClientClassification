import os
import sys
import numpy as np
import pickle
import pandas as pd
import random
import argparse

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

'''
Script to return training and testing word vectors
Expeced file structure: in working directory
 1) embeddings folder with glove.6B.100d.txt
 2)folder data with the training data.
'''

BASE_DIR = os.getcwd()
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
EMBEDDING_TYPE="embeddings" #glove

EMBEDDINGS_FILE= "glove.6B.100d.txt"
EMBEDDING_DIM = 100 #0 if character embeddings, 50 for above example

MODEL_DIR = BASE_DIR + EMBEDDING_TYPE
TEXT_DATA_DIR =  BASE_DIR + '/data/'

# number of words an entity is allowed to have
MAX_SEQUENCE_LENGTH = 10

# we use an egregious hack to check if its UTF-8
# Total number of unique tokens in company names is 37K
# Assuming no overlap between the two we get about 127K.  We may need to tweak this parameter as we go
# but according to the Keras documentation, this can even be left unset
MAX_NB_WORDS = 80000

VALIDATION_SPLIT = 0.2

# hack to check if the label is english
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

#function to return labeled texts
def label_data(TEXT_DATA_DIR, data_type):

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id - here the label name is just the name of the file in the data dir
    labels = []  # list of label ids

    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        if data_type in name:
            label_id = 0
            labels_index[data_type] = 0
        else:
            label_id=1
            labels_index['not_'+data_type]=1

        fpath = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(fpath):
            continue
        if sys.version_info < (3,):
            f = open(fpath)
        else:
            f = open(fpath, encoding='latin-1')

        for t in f.readlines():
            if not is_english(t):
                continue
            if (t.strip() == ''):
                continue
            num_tokens = len(t.strip().split(sep=' '))
            if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
                texts.append(t)
                labels.append(label_id)
        f.close()
    return texts,labels_index,labels

def process_texts(data_type):
    texts,labels_index,labels= label_data(TEXT_DATA_DIR, data_type)

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    save_path=BASE_DIR+ "/Models/lstm/3/" + data_type
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer_file = open(save_path+"/"+data_type+'_tokenizer.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(tokenizer, tokenizer_file)
    tokenizer_file.close()




    # this step creates a sequence of words ids for each word in each label
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a tiiraining set and a validation set
    # consider storing these indices/freezing them in order to accurately compare results
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=VALIDATION_SPLIT, random_state=42)
    return (X_train, y_train),(X_test,y_test)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", help="data type being processed")
    args = parser.parse_args()
    (X_train, y_train), (X_test, y_test)=process_texts(args.data_type)
    print((X_train, y_train))

if __name__ == "__main__":
    main()

