import os
import pandas as pd
import sys
import tempfile
import urllib

BASE_DIR ="C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\openaddr-collected-us_northeast\\us"

if __name__ == "__main__":
   frame = pd.DataFrame()
   list_ = []
   for root, dirs, filenames in os.walk(BASE_DIR):
        for f in filenames:
            if f.endswith("csv"):
                df = pd.read_csv(os.path.join(root, f),index_col=None, header=0, dtype='str')
                list_.append(df)
                break
            
   frame = pd.concat(list_)
   addresses= frame[['NUMBER', 'STREET']].apply(lambda x: ' '.join(str(x)), axis=1)
   addresses.to_csv("output_addresses.csv")
    
 