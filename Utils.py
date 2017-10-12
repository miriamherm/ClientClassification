import pickle
import numpy as np
import keras.preprocessing.text as text
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
import sys

from keras.preprocessing.sequence import pad_sequences


# use a language detector instead...
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def getVector(file, tokenizer, max_seq_length):
    vec = read_data(file, max_seq_length)
    vec = tokenizer.texts_to_sequences(vec)
    vec = pad_sequences(vec, maxlen=max_seq_length)
    return vec


def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def read_data(file, max_seq_length):

    texts = []  # list of text samples
    if sys.version_info < (3,):
        f = open(file)
    else:
        f = open(file, encoding='latin-1')
    for t in f.readlines():
        process(texts, t, max_seq_length)
    f.close()
    return texts


def process(texts, t, max_seq_length):
    if not is_english(t):
        return
    if (t.strip() == ''):
        return
    num_tokens = len(t.strip().split(sep=' '))
    if 0 < num_tokens < max_seq_length:
        texts.append(t)


# loads a set of embeddings
def word_embeddings(embedding_file):
    embeddings_index = {}
    f = open(embedding_file, encoding="utf8")
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print ("Unable to embed line \n")
    f.close()
    return embeddings_index


# loads a model, and associated tokenizer from a prefix
def load(file_prefix):
    print(file_prefix + 'nermodel.h5')
    json_file = open(file_prefix + 'nermodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(file_prefix+"nermodel.h5")

    outfile = file_prefix + "tokenizer.pkl"
    with open(outfile, 'rb') as pickle_file:
        tokenizer = pickle.load(pickle_file)
    return model, tokenizer

