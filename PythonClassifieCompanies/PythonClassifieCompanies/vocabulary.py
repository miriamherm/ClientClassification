import os
import pandas as pd
import sys
import tempfile
import urllib

def read_column(f):
   #indir = '/home/des/test'
    frame = pd.read_csv(
                f,
                encoding='utf-8',
                # sep='|', default , seperator
               # dtype=str,
            )
    count=frame['test_count'] > 1    
    frame=frame[count]
    return frame['word']

 
if __name__ == "__main__":
    dir=sys.argv[1]
    s1 = pd.Series(['ST'])
    for root, dirs, filenames in os.walk(dir):
        for f in filenames:
            words=read_column(os.path.join(root, f))
            s1=s1.append(words).drop_duplicates()
    s1.to_csv("address_vocabulary.csv")