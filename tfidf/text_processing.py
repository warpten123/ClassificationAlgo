from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import re
import numpy as np
import math


def manual_tokenization(text):
    container = ""
    newText = []
    for i in range(len(text)):
        if (text[i] != ' ' and text[i] != '\t'):
            container = container + text[i]
            if (i == len(text)-1):
                newText.append(text[i])
        else:
            newText.append(container)
            container = ""

    return newText


def toLowerCase(text):
    text = [word.lower() for word in text]
    return text


def removeStopWords(text):
    with open('tfidf/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    text = [word for word in text if word not in stop_words]
    return text


def removeSpecialCharacters(text):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', text)


def cleanRawFrequency(termFrequency={}):
    fromStopWords = removeStopWords(termFrequency.keys())
    for k in tuple(termFrequency.keys()):
        if k not in fromStopWords:
            del termFrequency[k]

    return termFrequency
