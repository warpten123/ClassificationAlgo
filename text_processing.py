from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import docx2txt  # lib for reading docx files
import re
import pandas as pd
import numpy as np
import PyPDF2
import pdfplumber

import csv

import math


def extractNames():
    print("extracting names")


def extractResearchTitle():
    print("extract research title")


def extractDepartment():
    print("extract department")


# texts_from_file = docx2txt.process("Introduction.docx")
# print(texts_from_file)


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


def text_tokenization(text):
    text = texts_from_file.split()
    return text


def toLowerCase(text):
    text = [word.lower() for word in text]
    return text


def removeStopWords(text):
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    text = [word for word in text if word not in stop_words]
    return text


def removeSpecialCharacters(text):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', text)


def save_to_file(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def stemming(processedText):
    # manual stemming
    # if word ends with ing or ed, ly, s, es, ion, er, ness, ful, al, ment, ist, ness, ness, remove it
    # wtf
    for i in range(len(processedText)):
        if (processedText[i].endswith('ing') or processedText[i].endswith('ed')):
            processedText[i] = processedText[i][:-3]
        elif (processedText[i].endswith('ly')):
            processedText[i] = processedText[i][:-2]
        elif (processedText[i].endswith('s')):
            processedText[i] = processedText[i][:-1]
        elif (processedText[i].endswith('es')):
            processedText[i] = processedText[i][:-2]
        elif (processedText[i].endswith('ion')):
            processedText[i] = processedText[i][:-3]
        elif (processedText[i].endswith('er')):
            processedText[i] = processedText[i][:-2]
        elif (processedText[i].endswith('ness')):
            processedText[i] = processedText[i][:-4]
        elif (processedText[i].endswith('ful')):
            processedText[i] = processedText[i][:-3]
        elif (processedText[i].endswith('al')):
            processedText[i] = processedText[i][:-2]
        elif (processedText[i].endswith('ment')):
            processedText[i] = processedText[i][:-4]
        elif (processedText[i].endswith('ist')):
            processedText[i] = processedText[i][:-3]
        elif (processedText[i].endswith('ness')):
            processedText[i] = processedText[i][:-4]
        elif (processedText[i].endswith('ness')):
            processedText[i] = processedText[i][:-4]
    return processedText


def term_frequency_calculation(processedText):
    # term frequency calculation
    # calculate the frequency of each word in the text
    # store the word and its frequency in a dictionary
    # return the dictionary
    term_frequency = {}
    for word in processedText:
        if word in term_frequency:
            term_frequency[word] += 1
        else:
            term_frequency[word] = 1
    return term_frequency


def compute_tf(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def removeNumbers(diction={}):
    for k in tuple(diction.keys()):
        if k.isdigit():
            del diction[k]
    return diction
# def sentence_creation(text):



def cleanRawFrequency(termFrequency={}):
    fromStopWords = removeStopWords(termFrequency.keys())
    for k in tuple(termFrequency.keys()):
        if k not in fromStopWords:
            del termFrequency[k]

    return termFrequency


def computeIDF(term_frequency):
    N = len(term_frequency)
    print(N)
    idf_dict = {}

    idf_dict = dict.fromkeys(term_frequency[0].keys(), 0)
    # print(idf_dict)

    # loop in term_frequency
    for document in term_frequency:
        for word, val in document.items():
            if val > 0:
                idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N / float(val))
    try:
        word_occurance = term_frequency[word] + 1
    except:
        word_occurance = 1
        return np.log(2/word_occurance)

    print(idf_dict)

    # set the

# text,processedText = [],[]
# fromPDF = PDFProcessing()
# n_docs = len(text) # number of text in text

# print("Initial Count of Words in File: " + str(n_docs))
# processedText = removeSpecialCharacters(fromPDF)
# processedText = toLowerCase(processedText)
# processedText = manual_tokenization(processedText)
# processedText = removeStopWords(processedText)
# # processedText = removeNumbers(processedText)
# print("After Preprocessing: " + str(len(processedText)))
# n_words_set = len(processedText) # number of unique words in text

# save_to_file("processedText.txt", processedText)
# term_frequency = term_frequency_calculation(processedText)
# # term_frequency = {k: v for k, v in term_frequency.items() if not k.isdigit()}
# print(term_frequency)


# compute_tf = compute_tf(term_frequency, processedText)


# print()
# idfs = computeIDF([term_frequency])
# print(idfs)


# processedText = stemming(processedText)
# term_frequency = term_frequency_calculation(processedText)
# print(calculate_tfidf(term_frequency, processedText))
# print(matching_score(processedText, processedText))
# print(processedText)
# processedText = text_tokenization(processedText)


# def calculate_tfidf(term_frequency, processedText):
#     # calculate tf-idf
#     # tf = term frequency
#     # idf = inverse document frequency
#     # tf-idf = tf * idf
#     # store the word and its tf-idf in a dictionary
#     # return the dictionary
#     tf_idf = {}
#     for word in term_frequency:
#         tf = term_frequency[word]
#         idf = processedText.count(word)
#         tf_idf[word] = tf * idf
#     return tf_idf

# def matching_score(query, processedText):
#     # calculate the matching score
#     # store the word and its matching score in a dictionary
#     # return the dictionary
#     matching_score = {}
#     for word in query:
#         matching_score[word] = processedText.count(word)
#     return matching_score


# lowercase the content from text


# check for repeating words in text and save it as a map<word,count>
# text = dict([(word, text.count(word)) for word in set(text)])
# print(text)
# print(str(len(text)))
