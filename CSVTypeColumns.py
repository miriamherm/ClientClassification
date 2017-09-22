import testmodel_Binary

import pandas as pd
import argparse
import numpy as np


def type_columns(file):
    try:
        df = pd.read_table(file, sep='\t')
    except:
        df = pd.read_table(file, sep=",")

    col_to_types = {}
    for data_type in testmodel_Binary.data_types:
        loaded_model, tokenizer=testmodel_Binary.load_model(testmodel_Binary.MODEL_DIR+data_type+"/", data_type)

        for c in df.columns:
            if df[c].dtype != np.object:
                continue
            #print(c)
            #print(df[c].dtype)
            texts = df[c].tolist()
            print(texts)

            if c in col_to_types:
                preds = col_to_types[c]
            else:
                preds = {}
                col_to_types[c] = preds
            sequences = tokenizer.texts_to_sequences(texts)
            data = testmodel_Binary.pad_sequences(sequences, maxlen=testmodel_Binary.MAX_SEQUENCE_LENGTH)
            #print(data_type)
            #print(loaded_model.predict(data)[:,0])
            preds[data_type] = np.average(loaded_model.predict(data)[:,0])

    for k in col_to_types:
        print(df[k].tolist())
        print(k, col_to_types[k])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to csv file to be typed")

    args = parser.parse_args()
    print(args.file)
    type_columns(args.file)


if __name__ == "__main__":
    main()
