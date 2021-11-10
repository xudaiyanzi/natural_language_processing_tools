import pandas as pd
import re
import string
import unicodedata

import contractions

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
stop_words.extend(['said', 'would', 'subject', 'use', 'also', 'like'])
stop_words = set(stop_words)


### NOTE!!!!!
### This code is structured this way for demonstration purposes and could be heavily optimized

def tokenize_text(text):
    """Deconstruct contractions and return tokenized list of words"""
    text = contractions.fix(text)
    words = word_tokenize(text)
    return words

def to_lowercase(text):
    """Lowercase all words"""
    return text.lower().strip()

def remove_punctuation(tokens):
    """Remove punctuation from list of tokenized words"""
    table = str.maketrans('', '', string.punctuation)
    clean_tokens = [w.translate(table) for w in tokens]
    return clean_tokens

def remove_stopwords(token, min_word_len):
    """Remove stopwords and short words from list of tokenized words"""
    if token not in stop_words and len(token)>=min_word_len:
        return token

def remove_non_ascii(token):
    """Remove non-ASCII characters from list of tokenized words"""
    new_word = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_word

def lemmatize_words(token):
    """Normalizes variations of tokens through lemmatization using WordNet"""
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(token)
    return lemma

def stem_words(token):
    """Normalizes variations of tokens through stemming"""
    ps = PorterStemmer()
    stem = ps.stem(token)
    return stem

def clean_text(text, min_word_len=4):
    """Master function to pass text through preprocessing stages"""
    if type(text) is str:
        clean_words = []
        words = to_lowercase(text)
        words = tokenize_text(words)
        words = remove_punctuation(words)
        for w in words:
            w = remove_non_ascii(w)
            if remove_stopwords(w, min_word_len) is not None:
                w = lemmatize_words(w)
                w = stem_words(w)
                clean_words.append(w)
        return " ".join(clean_words)
    
    
def getTopCoefs(num_terms, model, class_labels, feature_names):

    # Pull the coefficient values for the regression
    coef_vals = model.coef_

    # set a few variables
    tmp_weights=[]
    terms=[]
    group = []
    vals = []

    # loop through the coefficient values
    for i in range(len(coef_vals)):
        weights = list(coef_vals[i])

        # Reverse sort the coefficient weights and pull out the top values
        max_coef_weights = sorted(weights, reverse=True)[:num_terms]

        # Append the label name to a list
        group.append(class_labels[i])

        # Within each label group, loop through the top coefficients
        for j in max_coef_weights:
            coef = weights.index(j)

            # Create a list of the label name and the cofficient name
            terms.append(group[i] + " - " + feature_names[coef])

            # Within each coefficient, pull the weights for the other labels
            for k in range(len(coef_vals)):
                other_weights = list(coef_vals[k])

                #Create a list of the weights (rounded to 3 decimal places)
                tmp_weights.append(round(other_weights[coef],3))

            # append the list of weights to a list of weight lists
            vals.append(tmp_weights)

            # before moving to the next coefficient, reset the weights list
            tmp_weights=[]

    # I decided to use matplotlib to create my table of coeffient weights
    # To do this, I turned off all of the actual plot lines
    ax = plt.subplot(frame_on=False)

    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    table = plt.table(cellText=vals,
              colLabels=group,
              rowLabels=terms,
              loc='top'
             )
    # increase font size
    table.set_fontsize(20)
    table.scale(2,2)
    plt.show()