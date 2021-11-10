import pandas as pd
import re
import string
import unicodedata

import contractions

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn

stop_words = stopwords.words('english')
stop_words.extend(['said', 'would', 'subject', 'use', 'also', 'like'])
stop_words = set(stop_words)


def fix_contractions(text):
    """Deconstruct contractions and return tokenized list of words"""
    return contractions.fix(text)

def to_lowercase(text):
    """Lowercase all words"""
    return text.lower()

def remove_punctuation_and_numbers(text):
    """Remove punctuation from list of tokenized words"""
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation+string.digits)
    clean_tokens = [w.translate(table) for w in tokens]
    
    # remove empty values left behind from tokens that were only punctuation
    clean_tokens = list(filter(None, clean_tokens))
    return clean_tokens

def remove_stopwords(token, min_word_len):
    """Remove stopwords and short words from list of tokenized words"""
    if token not in stop_words and len(token)>=min_word_len:
        return token

# def remove_non_ascii(tokens):
#     """Remove non-ASCII characters from list of tokenized words"""
#     new_words = []
#     for word in tokens:
#         new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#         new_words.append(new_word)
#     return new_words

def lemmatize_words(token):
    """Normalizes variations of tokens through lemmatization using WordNet"""
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token)

def stem_words(token):
    """Normalizes variations of tokens through stemming"""
    ps = PorterStemmer()
    return ps.stem(token)

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def clean_text(text, min_word_len=4):
    """Master function to pass text through preprocessing stages"""
    if type(text) is str:
        clean_words = []
        words = remove_punctuation_and_numbers(text)
        for w in words:
            if remove_stopwords(w, min_word_len) is not None:
                clean_words.append(w)
        return " ".join(clean_words)