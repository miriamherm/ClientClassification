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




BASE_DIR = 'C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\'
# directory containing glove encodings from Wikipedia (we can swap this out for another encoding later)
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
GLOVE_DIR = BASE_DIR + 'glove'
TEXT_DATA_DIR = BASE_DIR + 'nerData'
TEST_DATA_DIR = BASE_DIR + 'testNERData'
PRED_DATA_DIR = BASE_DIR + 'CompanyClassifier'

# number of words an entity is allowed to have
# distribution of number of words in peoples names can be found in peopleNamesDisbn
# distribution of number of words in company names can be found in companyNamesDisbn
# Note most of the names above that are fairly esoteric or just plain noise.  Included is
# python code to remove them
MAX_SEQUENCE_LENGTH = 10


# hack to check if the label is english
def is_english(data):
    try:
        data.encode('utf-8')
    except UnicodeDecodeError:
        return False
    return True
def label_data(TEXT_DATA_DIR):
    
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id - here the label name is just the name of the file in the data dir
    labels = []  # list of label ids

    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        label_id = len(labels_index)
        labels_index[name] = label_id
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

def return_column_list(url):
    '''
    download column to classify.
    '''
    frame = pd.read_csv(
        url,
        encoding='utf-8',
        # sep='|', default , seperator
        dtype=str,
    )

    # Return specific column
   
    return list(frame)

def download_test_column(url,columnname):
    '''
    download column to classify.
    '''
    frame = pd.read_csv(
        url,
        encoding='utf-8',
        # sep='|', default , seperator
        dtype=str,
    )

    # Return specific column
    return frame[columnname]

# Total number of unique tokens in peoples names is 90K, including a lot of non-English names.  To remove those
# we use an egregious hack to check if its UTF-8
# Total number of unique tokens in company names is 37K
# Assuming no overlap between the two we get about 127K.  We may need to tweak this parameter as we go
# but according to the Keras documentation, this can even be left unset
MAX_NB_WORDS = 80000
# Size of embeddings from Glove (we will try the 100 dimension encoding to start with)
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the glove embeddings set
# to their embedding vector.  This is a straightforward lookup of
# words in Glove and then their embeddings which should be a 100 sized array of floats

'''
print('Reading word embeddings: Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts,labels_index,labels= label_data(TEXT_DATA_DIR)

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
num_words = min(MAX_NB_WORDS, len(word_index))
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
# it may be for when there are multiple layers to the network
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

tokenizer_file = open('tokenizer.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(tokenizer, tokenizer_file)
tokenizer_file.close()
'''

#test load from pickle from tokenizer
json_file = open('nermodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("nermodel.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tokenizer=None
tokenizer_file = open('tokenizer.pkl', 'rb')
tokenizer = pickle.load(tokenizer_file)

texts,labels_index,labels= label_data(TEST_DATA_DIR)

sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_test = data
y_test = labels

score= loaded_model.evaluate(x_test, y_test)
print(score)
score= loaded_model.predict(x_test, batch_size=167)
#concatenate text list with prediction
ls=np.asarray([w.replace('\n', '') for w in texts])
data_score=np.concatenate((ls[:,None],score),axis=1)
np.savetxt('model_predic.csv', data_score, fmt="%s", delimiter=',')   # X is an array


'''
def clean_column(column_data, column):
    noblanks = column_data != ""
    isEnglish= column_data.apply(is_english)
    num_tokens= (column_data.str.strip().split(sequence_input=' ')<MAX_SEQUENCE_LENGTH) and  (column_data.str.strip().split(sequence_input=' ')>0)
    column_data = column_data[noblanks]
    return column_data.values.tolist()
'''

def clean_column(column_data, column):
     a=column_data.as_matrix()
     #insert numpy cleansing functions
     #get rid of blanks and word sequences that are longer than MAX_SEQUENCE_LENGTH. 
     #all upper case
     return column_data
df=None
for name in sorted(os.listdir(PRED_DATA_DIR)):
    if not name.endswith(".csv"):
        continue
    fpath = os.path.join(PRED_DATA_DIR, name)
    if os.path.isdir(fpath):
        continue
    if sys.version_info < (3,):
        f = open(fpath)
    else:
        column_list=return_column_list(fpath)
        for column in column_list:
            texts=[]
            column_data=download_test_column(fpath, column)
            #make sure column is not just numbers
            if df is None:
                df=pd.DataFrame(column_data)
            else:
                df[column]=column_data
            texts=clean_column(column_data, column)
            sequences = tokenizer.texts_to_sequences(texts)
            data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            #dynamically generate batch_size based on column length
            results= loaded_model.predict(data, batch_size=51, verbose=1) 
            df[column+"_results"]=results[:,0]



'''
 serialize model to JSON
model_json = model.to_json()
with open("nermodel.json", "w") as json_file:
    json_file.write(model_json)
 serialize weights to HDF5
model.save_weights("nermodel.h5")
print("Saved model to disk")
'''

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