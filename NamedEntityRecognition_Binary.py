"""
This code is modified from
https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
for our own purposes
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, model_from_json
from keras.utils import to_categorical, plot_model

import enchant #spell checker library to check if words are english

#word files must be saved in base directory "data" folder
#first argument base directory for project
#second argument pre trained embedding folder
#third argument word/char embeddings flag
#fourth argument embedding file, if .npy character embedding, vocab.txt file must be saved in same directory
#fifth argument embedding dimension (0 if .NPY character embeddings)

#EXAMPLE: python NamedEntityRecognition_Binary.py C:\Users\Miriam\Documents\MastersResearch\DataScience\ glove word glove.6B.50d.txt 50
BASE_DIR = os.getcwd()
# directory containing glove encodings from Wikipedia (we can swap this out for another encoding later)
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
EMBEDDING_TYPE=sys.argv[1] #glove
WORD_CHAR_EMBEDDING= sys.argv[2] #word/char

EMBEDDINGS_FILE= sys.argv[3] #glove.6B.50d.txt
EMBEDDING_DIM = int(sys.argv[4]) #0 if character embeddings, 50 for above example

MODEL_DIR= BASE_DIR +"/"+ EMBEDDING_TYPE
TEXT_DATA_DIR =  BASE_DIR + '/data/'
EXP_NUM=sys.argv[5] #folder for experiment
# number of words an entity is allowed to have
MAX_SEQUENCE_LENGTH = 10

# we use an egregious hack to check if its UTF-8
# Total number of unique tokens in company names is 37K
# Assuming no overlap between the two we get about 127K.  We may need to tweak this parameter as we go
# but according to the Keras documentation, this can even be left unset
MAX_NB_WORDS = 80000

d= enchant.Dict("en_US")
VALIDATION_SPLIT = 0.2

# hack to check if the label is english
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True

def english_word(data):
#    print(data)
    try:
        d.check(data)
    except enchant.errors.Error:
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
            if not english_word(t):
                continue
            if (t.strip() == ''):
                continue
            num_tokens = len(t.strip().split(sep=' '))
            if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
                texts.append(t)
                labels.append(label_id)
        f.close()
    return texts,labels_index,labels

#function to read character embeddings
def read_data(TEXT_DATA_DIR, name):

    texts = []  # list of text samples
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(fpath):
        return
    if sys.version_info < (3,):
        f = open(fpath)
    else:
        f = open(fpath, encoding='utf-8')
        #f = open(fpath, encoding='latin-1')

    for t in f.readlines():
        if not is_english(t):
            continue
        if (t.strip() == ''):
            continue
        if (t.strip() == '!!!MAXTERMID'):
            continue
        num_tokens = len(t.strip().split(sep=' '))
        if 0 < num_tokens < MAX_SEQUENCE_LENGTH:
            texts.append(t.rstrip("\n"))
    f.close()
    return texts

#return embeddings_index from character embeddings
def char_embeddings():

    embeddings_char_cnn=np.load(os.path.join(MODEL_DIR, EMBEDDINGS_FILE))
    vocab=read_data(os.path.join(MODEL_DIR, "vocab.txt"))

    # Size of embeddings
    EMBEDDING_DIM = np.shape(embeddings_char_cnn)[1]

    print('Reading word embeddings: Indexing word vectors.')

    embeddings_index = {}
    for i, word in enumerate(vocab):
        try:
            coefs = np.asarray(embeddings_char_cnn[i], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print ("Unable to embed line \n")

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index,EMBEDDING_DIM


#return embeddings_index from word embeddings
def word_embeddings():

    embeddings_index = {}
    f = open(os.path.join(MODEL_DIR, EMBEDDINGS_FILE), encoding="utf8")
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


#train binary model and store in folder structure -> BASE_DIR->Models->Embedding_type->data_type
def trainBinaryModel(data_type):
    # second, prepare text samples and their labels
    save_path=BASE_DIR+ "Models/"+ EMBEDDINGS_FILE + "/"+ EXP_NUM+"/"+ data_type
    print('Processing '+ data_type+' text dataset')

    texts,labels_index,labels= label_data(TEXT_DATA_DIR, data_type)

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

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will1111 be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Dense(128, activation='relu')(embedded_sequences)

    # don't understand why we need flatten here?
    # the labels are 2D, need to flatten them
    x = Flatten()(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #serialize model to JSON
    model_json = model.to_json()
    with open(save_path+"/"+data_type+"_nermodel.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights(save_path+"/"+data_type+"_nermodel.h5")
    print("Saved model to disk")

    tokenizer_file = open(save_path+"/"+data_type+'_tokenizer.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(tokenizer, tokenizer_file)
    tokenizer_file.close()


#beging processing
if WORD_CHAR_EMBEDDING=="char":
    embeddings_index,EMBEDDING_DIM=char_embeddings()
else:
    embeddings_index,EMBEDDING_DIM=word_embeddings()

print('Found %s word vectors.' % len(embeddings_index))

# first, build index mapping words in the glove embeddings set
# to their embedding vector.  This is a straightforward lookup of
# words in Glove and then their embeddings which should be a 100 sized array of floats

print('Reading word embeddings: Indexing word vectors.')
data_types=["products", "names", "companies", "address", "other", "city", "state"]
for data_type in data_types:
    trainBinaryModel(data_type)
# later...

# load json and create model

'''
X=data
Y=labels

json_file = open('nermodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("nermodel.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)

print(score)
'''
