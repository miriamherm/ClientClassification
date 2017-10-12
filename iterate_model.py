# -*- coding: utf-8 -*-
"""
m keras.layers import Dense, Input, Flatten
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
from keras.models import Model, model_from_json, Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers.merge import concatenate

np.random.seed(1)
#first argument base directory for project
#second argument model name folder, with subfolders with models for each of the data_types
#third argument experiment #
BASE_DIR = os.getcwd()
TEST_DATA_DIR= os.path.join(BASE_DIR, "trainingdata")
MODEL_DIR= os.path.join(BASE_DIR, "Models/glove.6B.100d.txt/7/")
EMBEDDINGS_FILE = "embeddings/glove.6B.100d.txt"
EMBEDDING_DIM=100
MAX_SEQUENCE_LENGTH = 10
MAX_NB_WORDS=80000
data_type =sys.argv[1]#, "city", "state", "other"] 
VALIDATION_SPLIT=.2

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**.5) + 1) if n % i == 0)))
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def label_data(TEXT_DATA_DIR):
    labels_index = {}  # dictionary mapping label name to numeric id - here the label name is just the name of the file in the data dir
    labels = []  # list of label ids
    texts = []  # list of text samples
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        if data_type in name:
            label_id = 0
            labels_index[data_type] = 0
        else:
            label_id=1
            labels_index['not_'+data_type]=1
        fpath = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(fpath):
            return
        if sys.version_info < (3,):
            f = open(fpath)
        else:
            f = open(fpath, encoding='latin-1')
        for t in f.readlines():
        #process(texts, t, label_id, labels)
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

def process(texts, t, label_id):
    if not is_english(t):
        return
    if (t.strip() == ''):
        return
    num_tokens = len(t.strip().split(sep=' '))
    if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
        texts.append(t)
        labels.append(label_id)

def word_embeddings():

    embeddings_index = {}
    f = open(os.path.join(BASE_DIR, EMBEDDINGS_FILE), encoding="utf8")
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print ("Unable to embed line \n")
    f.close()
    return embeddings_index,EMBEDDING_DIM

def load_model(model_dir, data_type):

    #load json model
    json_file = open(model_dir+data_type+'_nermodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir+data_type+"_nermodel.h5")
    #loaded_model.layers.pop()
    for layer in loaded_model.layers:
         layer.trainable=False
    #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(loaded_model.get_config())   #load tokenizer
    tokenizer=None
   # tokenizer_file = open(model_dir+data_type+'_tokenizer.pkl', 'rb')
   # tokenizer = pickle.load(tokenizer_file)

    return loaded_model, tokenizer

def main():
    preds = dict()
    preds2 = dict()
    embeddings_index,EMBEDDING_DIM=word_embeddings()
    loaded_model, tokenizer=load_model(MODEL_DIR+data_type+"/", data_type)    
#for name in sorted(os.listdir(TEST_DATA_DIR)):
    #    print("Loading data "+ name)
    texts, labels_index, labels = label_data(TEST_DATA_DIR)
    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    # this step creates a sequence of words ids for each word in each label
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    # consider storing these indices/freezing them in order to accurately compare results
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    layer1=loaded_model.layers.pop()
    x = loaded_model.output
    x = Dense(128, activation='relu', name='dense8')(x)
    output= Dense(len(labels_index), activation='softmax', name ="dense9")(x)
    # x = Dense(128, activation='relu')(x)
    #preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(input=loaded_model.input, output=output)

    #model.fit(x_train, y_train,
     #         batch_size=128,
     #         epochs=10,
     #         validation_data=(x_val, y_val))     
    new_model=Sequential()
    #new_model.add(loaded_model) #keep original architecture and weights
    new_model.add(model) #add new layer
    new_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    print(new_model.summary)
    print("fitting sequential model")
    new_model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=10, 
                  validation_data=(x_val, y_val)
                 )
        # new_model.compile(...)
        #    sequences = tokenizer.texts_to_sequences(texts)
    save_path=BASE_DIR+ "/Models/update/"+ data_type#    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    #serialize model to JSON
    model_json = new_model.to_json()
    with open(save_path+"/"+data_type+"_nermodel.json", "w") as json_file:
       json_file.write(model_json)
    #serialize weights to HDF5
    new_model.save_weights(save_path+"/"+data_type+"_nermodel.h5")
    print("Saved model to disk")

    tokenizer_file = open(save_path+"/"+data_type+'_tokenizer.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(tokenizer, tokenizer_file)
    tokenizer_file.close()
        

if __name__ == "__main__":
    main()
