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

BASE_DIR = 'C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\'
TEST_DATA_DIR = BASE_DIR + 'testNERData'

MAX_SEQUENCE_LENGTH = 10


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
print("Evaluation of test set metrics:", score)
#find correct batchsize for input data. (The middle factor) 
factors_data=list(factors(data.shape[0]))
batch_size=sorted(factors_data)[len(factors_data)//2]
print("batch_size=", batch_size)
score= loaded_model.predict(x_test, batch_size=batch_size)

#concatenate text list with prediction
ls=np.asarray([w.replace('\n', '') for w in texts])
print("saving predictions to model_predic.csv")
data_score=np.hstack([ls[:,None],score]) #np.concatenate((ls[:,None],score),axis=1)
np.savetxt('model_predic.csv', data_score, fmt="%s")   # X is an array

#confusion matrix
#from http://learnandshare645.blogspot.in/2016/06/feeding-your-own-data-set-into-cnn.html
print("computing confusion matrix")
Y_pred = loaded_model.predict(x_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

target_names = ['class 0(NAMES)', 'class 1(COMPANIEs)']
print("confusion report:")
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

