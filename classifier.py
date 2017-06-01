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
'''


import sys
import csv
import collections
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Remember to update the script for the new data when you change this URL
URL = "C:\\Users\\Miriam\\Documents\\MastersResearch\\DataScience\\CompanyClassifier\\companyRedirects.csv"

# =====================================================================

#Step 1: Download the data.
def download_data(numcols):
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


numcols=147 #number of columns  in datafile
df=download_data(numcols)

#split company names with many words into individual words (on space) and combine to one column
def clean_data(dataf):
    dataf= dataf.astype(str)
    words_df = dataf.apply(lambda x: pd.Series(x.split(' ')))
    words_df = words_df.stack().reset_index(drop=True)
    return words_df
 

words= df.stack().reset_index(drop=True).dropna()
words_clean =clean_data(words)
print('Data size', len(words))


# Step 2: Build the dictionary and replace rare words with UNK token.

def build_dataset(words):
  d=collections.Counter(words)
  df= pd.DataFrame.from_dict(d,orient='index').reset_index()
  df=df.rename(columns={'index':'word', 0:'count'})
  matters= df['count']>3
  noblanks= df['word']!=""
  df=df[matters & noblanks]
  
  df=df.sort_values(by='count', ascending=False)
  df=df.reset_index(drop=True)
  
  return df

count = build_dataset(words_clean)
del words, words_clean, df  # Hint to reduce memory.

count.to_csv("company_counts.csv")


#data_index = 0

## Step 3: Function to generate a training batch for the skip-gram model.
#def generate_batch(batch_size, num_skips, skip_window):
#  global data_index
#  assert batch_size % num_skips == 0
#  assert num_skips <= 2 * skip_window
#  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#  buffer = collections.deque(maxlen=span)
#  for _ in range(span):
#    buffer.append(data[data_index])
#    data_index = (data_index + 1) % len(data)
#  for i in range(batch_size // num_skips):
#    target = skip_window  # target label at the center of the buffer
#    targets_to_avoid = [skip_window]
#    for j in range(num_skips):
#      while target in targets_to_avoid:
#        target = random.randint(0, span - 1)
#      targets_to_avoid.append(target)
#      batch[i * num_skips + j] = buffer[skip_window]
#      labels[i * num_skips + j, 0] = buffer[target]
#    buffer.append(data[data_index])
#    data_index = (data_index + 1) % len(data)
#  # Backtrack a little bit to avoid skipping words in the end of a batch
#  data_index = (data_index + len(data) - span) % len(data)
#  return batch, labels

#batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
#for i in range(8):
#  print(batch[i], reverse_dictionary[batch[i]],
#        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

## Step 4: Build and train a skip-gram model.

#batch_size = 128
#embedding_size = 128  # Dimension of the embedding vector.
#skip_window = 1       # How many words to consider left and right.
#num_skips = 2         # How many times to reuse an input to generate a label.

## We pick a random validation set to sample nearest neighbors. Here we limit the
## validation samples to the words that have a low numeric ID, which by
## construction are also the most frequent.
#valid_size = 16     # Random set of words to evaluate similarity on.
#valid_window = 100  # Only pick dev samples in the head of the distribution.
#valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#num_sampled = 64    # Number of negative examples to sample.

#graph = tf.Graph()

#with graph.as_default():

#  # Input data.
#  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
#  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

#  # Ops and variables pinned to the CPU because of missing GPU implementation
#  with tf.device('/cpu:0'):
#    # Look up embeddings for inputs.
#    embeddings = tf.Variable(
#        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#    # Construct the variables for the NCE loss
#    nce_weights = tf.Variable(
#        tf.truncated_normal([vocabulary_size, embedding_size],
#                            stddev=1.0 / math.sqrt(embedding_size)))
#    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

#  # Compute the average NCE loss for the batch.
#  # tf.nce_loss automatically draws a new sample of the negative labels each
#  # time we evaluate the loss.
#  loss = tf.reduce_mean(
#      tf.nn.nce_loss(weights=nce_weights,
#                     biases=nce_biases,
#                     labels=train_labels,
#                     inputs=embed,
#                     num_sampled=num_sampled,
#                     num_classes=vocabulary_size))

#  # Construct the SGD optimizer using a learning rate of 1.0.
#  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

#  # Compute the cosine similarity between minibatch examples and all embeddings.
#  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
#  normalized_embeddings = embeddings / norm
#  valid_embeddings = tf.nn.embedding_lookup(
#      normalized_embeddings, valid_dataset)
#  similarity = tf.matmul(
#      valid_embeddings, normalized_embeddings, transpose_b=True)

#  # Add variable initializer.
#  init = tf.global_variables_initializer()

## Step 5: Begin training.
#num_steps = 100001

#with tf.Session(graph=graph) as session:
#  # We must initialize all variables before we use them.
#  init.run()
#  print("Initialized")

#  average_loss = 0
#  for step in xrange(num_steps):
#    batch_inputs, batch_labels = generate_batch(
#        batch_size, num_skips, skip_window)
#    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

#    # We perform one update step by evaluating the optimizer op (including it
#    # in the list of returned values for session.run()
#    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#    average_loss += loss_val

#    if step % 2000 == 0:
#      if step > 0:
#        average_loss /= 2000
#      # The average loss is an estimate of the loss over the last 2000 batches.
#      print("Average loss at step ", step, ": ", average_loss)
#      average_loss = 0

#    # Note that this is expensive (~20% slowdown if computed every 500 steps)
#    if step % 10000 == 0:
#      sim = similarity.eval()
#      for i in xrange(valid_size):
#        try:
#            valid_word = reverse_dictionary[valid_examples[i]]
#            top_k = 8  # number of nearest neighbors
#            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
#            log_str = "Nearest to %s:" % valid_word
#            for k in xrange(top_k):
#              close_word = reverse_dictionary[nearest[k]]
#              log_str = "%s %s," % (log_str, close_word)
#            print(log_str)
#        except UnicodeEncodeError:
#            print(UnicodeEncodeError.reason)
#  final_embeddings = normalized_embeddings.eval()

## Step 6: Visualize the embeddings.


#def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#  plt.figure(figsize=(18, 18))  # in inches
#  for i, label in enumerate(labels):
#    x, y = low_dim_embs[i, :]
#    plt.scatter(x, y)
#    plt.annotate(label,
#                 xy=(x, y),
#                 xytext=(5, 2),
#                 textcoords='offset points',
#                 ha='right',
#                 va='bottom')

#  plt.savefig(filename)

#try:
#  from sklearn.manifold import TSNE
#  import matplotlib.pyplot as plt

#  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#  plot_only = 500
#  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#  plot_with_labels(low_dim_embs, labels)

#except ImportError:
#    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
