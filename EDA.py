import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex
import re
from sklearn.model_selection import train_test_split
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

eng_stopwords = set(stopwords.words("english"))

df = pd.read_csv('train.csv')

x = df.iloc[:, 2:].sum()

# fig, ax = plt.subplots(nrows=1, ncols=1)

# ax.pie(x.values, labels = x.index, autopct="%1.1f%%")

# plt.show()

rowsums = df.iloc[:, 2:].sum(axis = 1)

df['clean_comment'] = (rowsums == 0)

dict_ex = {True : 1, False : 0}

df['clean_comment'] = df['clean_comment'].map(dict_ex)


df['letter_count'] = df.comment_text.apply(lambda x : len(str(x)))


# df['letter_count'].hist()

df['count_word'] = df['comment_text'].apply(lambda x : len(str(x).split()))
df['unique_word_count'] = df['comment_text'].apply(lambda x : len(set(str(x).split())))
df['unique_word_count'] = df['comment_text'].apply(lambda x : len(set(str(x).split())))
df['punctuation_count'] = df['comment_text'].apply(lambda x : len([c for c in str(x) if c in string.punctuation]))
df['upper_word_count'] = df['comment_text'].apply(lambda x : len([c for c in str(x).split() if c.isupper()]))
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([c for c in str(x).lower().split() if c in eng_stopwords]))
df['mean_word_length'] = df['comment_text'].apply(lambda x : np.mean([len(c) for c in str(x).split()]))