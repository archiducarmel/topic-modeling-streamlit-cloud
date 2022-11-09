import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from wordcloud import WordCloud

import nltk
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed


lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()

    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,
                                                             'a'))  # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatise adverbs
        else:
            lemmatized_text_list.append(
                lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatisation

    return " ".join(lemmatized_text_list)

def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])

def contraction_text(text):
    return contractions.fix(text)

negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"


def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i + 1 for i in range(len(tokens) - 1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx] = negative_prefix + tokens[idx]

    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]

    return " ".join(tokens)

def remove_stopwords(text):
    english_stopwords = stopwords.words("english")
    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    # Tokenize review
    text = tokenize_text(text)

    # Lemmatize review
    text = lemmatize_text(text)

    # Normalize review
    text = normalize_text(text)

    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)

    # Remove stopwords
    text = remove_stopwords(text)

    return text

ADJECTIVES_LIST = ['good',
                 'great',
                 'delicious',
                 'nice',
                 'friendly',
                 'little',
                 'fresh',
                 'first',
                 'favorite',
                 'excellent',
                 'small',
                 'awesome',
                 'perfect',
                 'amaze',
                 'hot',
                 'special',
                 'sweet',
                 'amazing',
                 'super',
                 'happy']