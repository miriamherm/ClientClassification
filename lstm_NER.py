import process_texts
import os 

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import sys

BASE_DIR=os.getcwd()
exp_num=sys.argv[1]
max_features = 80000
maxlen = 1  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
num_epochs=10

def create_model(data_type):
	print('Loading data...')
	(x_train, y_train), (x_test, y_test) = process_texts.process_texts(data_type)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

#print('Pad sequences (samples x time)')
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
	print('x_train shape:', x_train.shape)
	print('y_train shape:', y_test.shape)

	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(2, activation='sigmoid'))

# try using different optimizers and different optimizer configs
	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(x_test, y_test))
	score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

	model_json = model.to_json()

	save_path=BASE_DIR+ "/Models/lstm/"+ exp_num+"/" +  data_type
	with open(save_path+"/"+data_type+"_nermodel.json", "w") as json_file:
        	json_file.write(model_json)
    #serialize weights to HDF5
	model.save_weights(save_path+"/"+data_type+"_nermodel.h5")
	print("Saved model to disk")

data_types=[ "address","companies", "names", "products"]
for data_type in data_types:
	create_model(data_type)
