'''
This script perfoms the basic process for applying a classification
algorithm to a dataset using Python libraries.

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
import ntpath

import pandas as pd

from datetime import datetime

from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

URL=sys.argv[1]  #"C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\CompanyClassifier\\companyRedirects.csv"
URL_Test = sys.argv[2] #"C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\CompanyClassifier\\CompanyTest\\Prequalified_Firms.csv"
trainingfile = "training_company_counts.csv"

# =====================================================================

# Step 1: Download the data.
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
    del frame[0]  # wikipedia link column
    return frame


# Step 2: Build the dictionary and replace rare words with UNK token.
def build_training_dataset(words):
    d = collections.Counter(words)
    df = pd.DataFrame.from_dict(d, orient='index').reset_index()
    df = df.rename(columns={'index': 'word', 0: 'count'})
    matters = df['count'] > 3
    noblanks = df['word'] != ""
    df = df[matters & noblanks]
    df['norm'] = df['count'] / len(df['count'])
    df = df.sort_values(by='count', ascending=False)
    df = df.reset_index(drop=True)

    return df


# remove extra characters from company names, leaving "" or " " depdning on significance of symbol.
def clean_company_names(companies):
    # companies = companies.str.replace("(", " ")
    # companies = companies.str.replace(")", " ")

    words_df = companies.str.split(" ", expand=True)
    words = words_df.stack().reset_index(drop=True)
    stop_words = ['AND','OF','AT','IS','ON','FOR','THE']
    stop_chars=([' ','@','&','%','^','<','>','0','1','2','3','4','5','6','7','8','9','+','-','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

    char_list=words.apply(lambda x: [item for item in x if item not in stop_chars])
   
    word_comp=[''.join(map(str,ls)).upper() for ls in char_list if ls!=[]]
    word_list=[word for word in word_comp if len(word)>1 and word not in stop_words]

    #timed it, 2.342168s for list comprehension, 4.286556s for loop

    #word_list=list()
    #for ls in char_list:
    #    tmp=''.join(map(str,ls)).upper()
    #    if tmp!='' and len(tmp)>1 and tmp not in stop_words:
    #        word_list.append(tmp)

    return word_list


def train_data():
    numcols = 147  # number of columns  in datafile - columns are irregularly filled
    df = download_data_no_columns(numcols)
    companies = df.stack().reset_index(drop=True).dropna()  # make into one column of companies
    words = clean_company_names(companies)  # clean and format text into single words, capitalized

    trainingfile = build_training_dataset(words)  # calculate word frequency, and normalize
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


# same function as "build_training_dataset" except nothing is removed and naming conventions are different
def build_test_dataset(words):
    d = collections.Counter(words)
    df = pd.DataFrame.from_dict(d, orient='index').reset_index()
    df = df.rename(columns={'index': 'word', 0: 'test_count'})
    noblanks = df['word'] != ""
    df = df[noblanks]
    df['test_norm'] = df['test_count'] / len(df['test_count'])
    df = df.sort_values(by='test_count', ascending=False)
    df = df.reset_index(drop=True)

    return df

# return cosine similarity between testing datacolumn and training dataset
def test_column(URL_Test, columnName):

    companies = download_test_column(columnName).drop_duplicates()
    words = clean_company_names(companies)
    testing = build_test_dataset(words)

    return testing


def check_missing_words(testing, training):
    df = pd.merge(training, testing, on='word', how='outer')
    missing = df[df.isnull().any(axis=1)]
    matters = (missing['count'] > 3) | (missing['test_count']>1)
    
    missing=missing[matters]
    return missing


# train dataset and test outside data column on it
if __name__ == "__main__":
    # Remember to update the script for the new data when you change this URL
    test_file_column= sys.argv[3]
    test_file_name=ntpath.basename(URL_Test)

  
    # training = train_data()
    testing = test_column(URL_Test, test_file_column)
   # missing = check_missing_words(testing, training)
    
   # df = pd.merge(training, testing, on='word', how='inner')
    #sim= cosine(df['norm'], df['test_norm'])

   # training.to_csv(trainingfile)
    testingfile = "results\\"+"ADD_"+test_file_column+test_file_name
   # missingfile = "results\\missing_"+ test_file_name
    
    testing.to_csv(testingfile) 
  #  missing.to_csv(missingfile)
