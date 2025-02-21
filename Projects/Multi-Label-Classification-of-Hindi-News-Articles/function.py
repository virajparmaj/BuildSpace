import pandas as pd
import chardet
import numpy as np
from nltk.tokenize import word_tokenize
import math
import csv

# define punctuations list
punctuations = ['nn','n', '।','/', '`', '+', '"', '?', '_(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', '#', '*', '', ';', '-', '}','|','"','\\', "''", '``']



# Loadstopwords from "stopwords.csv"
def load_stopwords(csv_file):
    stopwords = []
    with open(csv_file,'rb') as file:
        result = chardet.detect(file.read())
        encoding=result['encoding']
    with open(csv_file, encoding=encoding) as file:
        reader = csv.reader(file)
        for row in reader:
            for word in row:
                if len(word.strip()) != 0:
                    stopwords.append(word.strip())
    return stopwords



# remove stopwords from 'tokens' list
def stopword(tokens,stopwords):
    t = 0
    while (t < len(tokens)):
        if tokens[t].strip() in stopwords or tokens[t].strip('। ') in stopwords or tokens[t].strip().isnumeric():
            item = tokens[t]
            c = tokens.count(item)
            for n in range(c):
                tokens.remove(item)
        else:
            tokens[t] = tokens[t].strip('। ')
        t += 1



# calculate tf
# tf(w) = (frequency of word 'w' in an article/ total words in the article) averaged over articles in a class
def calculate_tf(dataset,stopwords):
    cols = dataset.columns.ravel()
    labels_tf = []
    for i in range(3,17):
        label_articles = dataset[dataset[cols[i]] == 1]['desc']
        label = []
        for article in label_articles:
            # tokenize article
            tokens = word_tokenize(article)

            # remove stopwords from 'tokens' list
            stopword(tokens,stopwords)

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
    return labels_tf



# calculate 'idf' for each word in the document
# idf(w) = log(total words in corpus/ frequency of word 'w' in the corpus)
# used formula: idf(w) = log(total articles in corpus/ frequency of articles with word 'w' in the corpus)
def calculate_idf(dataset, stopwords):
    corpus_articles = dataset['desc']
    corpus = []
    word_count = 0
    for article in corpus_articles:
        # tokenize article
        tokens = list(set(word_tokenize(article))) # to get unique tokens
        word_count += len(tokens)

        # remove stopwords from 'tokens' list
        stopword(tokens, stopwords)

        # convert list of tokens to dataframe
        tokens_df = pd.DataFrame(data=tokens, columns=['token'])

        # add a count column and set value as '1'
        tokens_df['count'] = 1

        # append dataframe containing token count to the list 'corpus'
        corpus.append(tokens_df)

    # calculate idf for each word in the document
    idf_df = pd.concat(corpus, ignore_index=True).groupby(['token'], as_index=False).sum()

    # changing column name to 'idf' from 'count'
    idf = idf_df.copy()
    idf['count'] = np.log((len(corpus_articles)-idf['count']+0.5)/ (idf['count']+0.5))  #apply idf formula
    idf = idf.rename(columns={'count': 'idf'})
    return idf



# calulate (tf * idf) for each token of each label
def calculate_tf_idf(labels_tf, idf):
    tf_idf = []
    for label_tf in labels_tf:
        label_tf_itf = pd.merge(label_tf, idf, on='token')
        label_tf_itf['tf-idf'] = label_tf_itf['tf'] * label_tf_itf['idf']
        label_tf_itf = label_tf_itf.sort_values(by=['tf-idf'], ascending=False)
        tf_idf.append(label_tf_itf)
    return tf_idf



# store top 100 tokens for each label in different excel sheets in a output file (arranged in descending order of tf-itf score)
def store_features(tf_idf, cols, output_file):
    i = 3
    frames = {}
    for df in tf_idf:
        sheet = cols[i]
        frames[sheet] = df.head(100)
        i += 1

    with pd.ExcelWriter(output_file) as writer:
        for frame in frames:
            frames[frame].to_excel(writer, sheet_name=frame)



# calculate tf-idf score for an article
def score(article, label_features, cols, stopwords):
    tokens = word_tokenize(article)
    stopword(tokens, stopwords)
    scores = []
    for i in range(3, 17):
        sheet = cols[i]
        features = label_features[i-3]
        dic = {}
        top_tokens = features['token']
        for token in top_tokens:
            dic[token] = list(features[features['token']==token]['tf-idf'])[0]
        score = 0
        for token in tokens:
            if token in dic:
                score += dic[token]
        scores.append(score)
    return scores