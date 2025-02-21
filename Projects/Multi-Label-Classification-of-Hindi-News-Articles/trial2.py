# perform multi label classification of news articles
# a labelled dataset is available for training the model

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import math
import csv

# Load stopwords from a csv file and store them in a list
stopwords = []
with open("stopwords.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        for word in row:
            if len(word.strip()) != 0:
                stopwords.append(word.strip())

# punctuation list
punctuations = ['nn','n', '।','/', '`', '+', '"', '?', '_(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', '#', '*', '', ';', '-', '}','|','"','\\', "''", '``']
stopwords.extend(punctuations)

# get dataframe for the dataset
try:
    dataset = pd.read_excel("preprocessed_data_new.xlsx")
except FileNotFoundError as e:
    raise Exception(e)

# get column names(17 out of which latter 14 are class labels)
cols = dataset.columns.ravel()

# calculate tf
# tf(w) = (frequency of word 'w' in an article/ total words in the article) averaged over articles in a class
labels_tf = []
for i in range(3,17):
    label_articles = dataset[dataset[cols[i]] == 1]['desc']
    label = []
    for article in label_articles:
        # tokenize article
        tokens = word_tokenize(article)

        # remove stopwords from 'tokens' list
        t = 0
        while (t < len(tokens)):
            if tokens[t].strip() in stopwords or tokens[t].strip('। ') in stopwords or tokens[t].strip().isnumeric():
                item = tokens[t]
                c = tokens.count(item)
                for n in range(c):
                    tokens.remove(item)
            t += 1

        # convert list of tokens to dataframe
        tokens_df = pd.DataFrame(data=tokens, columns=['token'])

        # calculate count of each token(divided by article length) and stored in a series frame
        sf = tokens_df.value_counts() / len(tokens_df)

        # convert series frame to dataframe
        df = sf.to_frame().reset_index()
        df = df.rename(columns={0: 'tf'})

        # append dataframe containing token count to the list 'label'
        label.append(df)

    # calculate 'tf' for a label by summing over each articles for a label and taking average(divided by no of articles)
    df1 = pd.concat(label, ignore_index=True).groupby(['token'], as_index=False).sum()
    df1['tf'] = df1['tf'] / len(label)

    # append the dataframe containing 'tf' value for a label to a list
    labels_tf.append(df1)



# calculate 'idf' for each word in the document
# idf(w) = log(total words in corpus/ frequency of word 'w' in the corpus)
corpus_articles = dataset['desc']
corpus = []
word_count = 0
for article in corpus_articles:
    # tokenize article
    tokens = word_tokenize(article)
    word_count += len(tokens)

    # remove stopwords from 'tokens' list
    t = 0
    while (t < len(tokens)):
        if tokens[t].strip() in stopwords or tokens[t].strip('। ') in stopwords or tokens[t].strip().isnumeric():
            item = tokens[t]
            c = tokens.count(item)
            for n in range(c):
                tokens.remove(item)
        t += 1

    # convert list of tokens to dataframe
    tokens_df = pd.DataFrame(data=tokens, columns=['token'])

    # calculate count of each token and stored in a series frame
    sf = tokens_df.value_counts()

    # convert series frame to dataframe
    df = sf.to_frame().reset_index()
    df = df.rename(columns={0: 'count'})

    # fake (can remove this, but have to check change in accuracy)
    df['count'] = 1

    # append dataframe containing token count to the list 'corpus'
    corpus.append(df)

# calculate idf for each word in the document
idf_df = pd.concat(corpus, ignore_index=True).groupby(['token'], as_index=False).sum()

# changing column name to 'idf' from 'count'
idf = idf_df.copy()
idf['count'] = np.log(len(corpus_articles)/ idf['count'])  #word_count
idf = idf.rename(columns={'count': 'idf'})

# calulate (tf * idf) for each token of each label
tf_idf = []
for label_tf in labels_tf:
    label_tf_itf = pd.merge(label_tf, idf, on='token')
    label_tf_itf['tf-idf'] = label_tf_itf['tf'] * label_tf_itf['idf']
    label_tf_itf = label_tf_itf.sort_values(by=['tf-idf'], ascending=False)
    tf_idf.append(label_tf_itf)



# store words for each label in different excel sheets in a output file (arranged in descending order of tf-itf score)
i = 3
frames = {}
for df in tf_idf:
    sheet = cols[i]
    frames[sheet] = df.head(100)
    i += 1

with pd.ExcelWriter('output_new.xlsx') as writer:
    for frame in frames:
        frames[frame].to_excel(writer, sheet_name=frame)



