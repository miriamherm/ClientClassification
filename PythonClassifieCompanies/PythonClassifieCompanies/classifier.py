'''
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the data (using collections)
   3. Train and evaluate learners (using tensorflow)
   4. Plot and compare results (using matplotlib)


The data was downloaded from wikipedia, but is loaded through CSV here. 

Code here updated & edited, orginally from: Microsoft Visual Studio Python templates 
& https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
http://www.apache.org/licenses/LICENSE-2.0

Licensed under the Eclipse Public License, Version 1.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
https://www.eclipse.org/legal/epl-v10.html
'''


import sys
import csv
import collections
import math
import os
import pandas as pd

from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

# Remember to update the script for the new data when you change this URL
URL = "C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\CompanyClassifier\\companyRedirects.csv"
URL_Test="C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\CompanyClassifier\\CompanyTest\\Prequalified_Firms.csv"
trainingfile="training_company_counts.csv"
testingfile="testing_company_counts.csv"
# =====================================================================

#Step 1: Download the data.
def download_data_no_columns(numcols):
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''
    frame = pd.read_csv(
        URL,
        encoding='utf-8', 
        sep='|',            
        dtype=str,
        names=range(numcols),
    )

    # Return the entire frame
    del frame[0] #wikipedia link column
    return frame

# Step 2: Build the dictionary and replace rare words with UNK token.
def build_training_dataset(words):
  d=collections.Counter(words)
  df= pd.DataFrame.from_dict(d,orient='index').reset_index()
  df=df.rename(columns={'index':'word', 0:'count'})
  matters= df['count']>3
  noblanks= df['word']!=""
  df=df[matters & noblanks]
  df['norm']=df['count']/len(df['count'])
  df=df.sort_values(by='count', ascending=False)
  df=df.reset_index(drop=True)
  
  return df

#remove extra characters from company names, leaving "" or " " depdning on significance of symbol.
def clean_company_names(companies):
     words_df=companies.str.split(" ", expand=True)
     words=words_df.stack().reset_index(drop=True)
     words=words.str.replace(".","")
     words=words.str.replace("+","")
     words=words.str.replace("-","")
     words=words.str.replace("`","")
     words=words.str.replace("'","")
     words=words.str.replace("%","")
     words=words.str.replace("$","")
     words=words.str.replace("#","")
     words=words.str.replace(",","")
     words=words.str.replace("/","")
     words=words.str.replace("-","")
     words=words.str.replace("&","")
     
     words=words.str.replace("("," ")
     words=words.str.replace(")"," ")

     words=words.str.upper()
     return words

def train_data():
    numcols=147 #number of columns  in datafile - columns are irregularly filled
    df=download_data_no_columns(numcols)
    companies= df.stack().reset_index(drop=True).dropna() #make into one column of companies
    words=clean_company_names(companies) #clean and format text into single words, capitalized
  
    trainingfile= build_training_dataset(words) #calculate word frequency, and normalize
    return trainingfile


def download_test_column(columnname):
    '''
    download column to classify.
    '''
    frame = pd.read_csv(
        URL_Test,
        encoding='utf-8', 
       # sep='|', default , seperator            
        dtype=str,
    )

    # Return specific column 
    return frame[columnname]

#same function as "build_training_dataset" except nothing is removed and naming conventions are different
def build_test_dataset(words):
  d=collections.Counter(words)
  df= pd.DataFrame.from_dict(d,orient='index').reset_index()
  df=df.rename(columns={'index':'word', 0:'test_count'})
  noblanks= df['word']!=""
  df=df[noblanks]
  df['test_norm']=df['test_count']/len(df['test_count'])
  df=df.sort_values(by='test_count', ascending=False)
  df=df.reset_index(drop=True)
  
  return df

#return cosine similarity between testing datacolumn and training dataset
def test_column(URL_Test, columnName, trainingset ):
   
    training=trainingset
  
    companies=download_test_column(columnName).drop_duplicates()
    words=clean_company_names(companies)
    testing = build_test_dataset(words)
   
    df=pd.merge(training, testing, on='word', how='inner')
    print("1- cosine difference:" , 1- cosine(df['norm'], df['test_norm']))
    return testing

#train dataset and test outside data column on it
if __name__ == "__main__":
    training=train_data()
    testing=test_column(URL_Test,"Prequalified Vendor Name", training)

    training.to_csv(trainingfile)
    testing.to_csv(testingfile)