import Utils
import pickle
import numpy as np
import argparse
import os
from keras.optimizers import SGD
from keras.layers import Flatten, Dense, Input

from keras.models import Model, model_from_json

MAX_SEQUENCE_LENGTH = 10


# file_prefix is the model file (and associated tokenizer) prefix
# training_file is the new data to be used for training
# test_file is the data to be used for testing all classes
def main(file_prefix, training_file, test_file, freeze, save_path):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    model, tokenizer = Utils.load(file_prefix=os.path.join(script_dir, file_prefix))

    train = Utils.getVector(os.path.join(script_dir, training_file), tokenizer, MAX_SEQUENCE_LENGTH)

    test = Utils.getVector(os.path.join(script_dir, test_file), tokenizer, MAX_SEQUENCE_LENGTH)

    print("Existing training types accuracy:")
    predict = model.predict(train)
    print(np.average(predict[:, 0]))
    print("Existing trained types accuracy:")
    predict =  model.predict(test)
    print(np.average(predict[:, 0]))

    # freeze all layers except the connection from the hidden layer to the output layer
    if freeze == 'True':
        for layer in  model.layers[:4]:
            print(layer)
            print('set to not trainable')
            layer.trainable = False
        for layer in  model.layers[4:]:
            print(layer)
            print('set to trainable')
            layer.trainable = True


    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    rows = np.shape(train)[0]
    arr = []
    for i in range(rows):
        arr.append([0, 1])

    y = np.asarray(arr)
    print(y)

    print(np.shape(y))
    print(np.shape(train))

    model.fit(train, y,
                  batch_size=128,
                  epochs=500)

    print("After training: Existing training types accuracy:")
    predict = model.predict(train)
#    print(predict[:, 0])
    print(np.average(predict[:, 0]))
    print("After training: Existing trained types accuracy:")
    predict = model.predict(test)
    print(np.average(predict[:, 0]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_json = model.to_json()
   # print(file_prefix.split("/")[-1])
    with open(save_path+"/"+file_prefix.split("/")[-1]+"nermodel.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights(save_path+"/"+file_prefix.split("/")[-1]+"nermodel.h5")
    tokenizer_file = open(save_path+"/"+file_prefix.split("/")[-1]+'tokenizer.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(tokenizer, tokenizer_file)
    tokenizer_file.close()
    print("Saved model to disk")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file_prefix')
    parser.add_argument('--train_file')
    parser.add_argument('--test_file')
    parser.add_argument('--freeze')
    parser.add_argument('--save_path')

    args = parser.parse_args()

    main(args.file_prefix, args.train_file, args.test_file, args.freeze, args.save_path)
