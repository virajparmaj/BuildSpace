# perform multi label classification of news articles
# a labelled dataset is available for training the model

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import math
import csv
from sklearn.model_selection import train_test_split
from function import punctuations
from function import load_stopwords, stopword, calculate_tf, calculate_idf, calculate_tf_idf, store_features

# Load stopwords from a csv file and store them in a list
stopwords = load_stopwords("stopwords.csv")

# add punctuations in the list of stopwords
stopwords.extend(punctuations)

# get dataframe for the dataset
try:
    data = pd.read_excel("preprocessed_data_new.xlsx")
except FileNotFoundError as e:
    raise Exception(e)

# get column names(17 out of which latter 14 are class labels)
cols = data.columns.ravel()
print(data.shape)

# dividing dataset into training and testing modules
X = data[cols[:3]]
y = data[cols[3:17]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Concatenate X_train and y_train into a single dataframe
train_df = pd.concat([X_train, y_train], axis=1)
#label_cols = cols[3:]
#label_freq = train_df[label_cols].sum()
#print(label_freq)
test_df = pd.concat([X_test, y_test], axis=1)
#label_freq = test_df[label_cols].sum()
#print(label_freq)
#exit()

# initialise training module as dataset to perform feature extraction on it using tf-idf method
dataset = train_df

# calculate tf
# returnns a list of dataframes with each df consisiting tf values of tokens of a label
labels_tf = calculate_tf(dataset, stopwords)

# calculate 'idf' for each word in the document
# returns a dataframe consisting of idf values of every token in the document
idf = calculate_idf(dataset, stopwords)

# calulate (tf * idf) for each token of each label
# return a list of dataframes consisting of tf, idf, and tf*idf values for each token for a label
tf_idf = calculate_tf_idf(labels_tf, idf)

# store top 100 tokens for each label in different excel sheets in a output file (arranged in descending order of tf-itf score)
store_features(tf_idf, cols, "output_new1.xlsx")
