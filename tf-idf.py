import text_processing as preProc
import pdfplumber
import re
import csv
import pandas as pd
import glob
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os
import math
import numpy as np
# class word:
#     def __init__(self, word, count):
#         self.word = word
#         self.count = count


class Processing():
    def __init__(self, path):
        self.path = path

    def getFromPDF(self):
        finalText = " "
        with pdfplumber.open(r'../Data Set/Goal 2/goal 2_3.pdf') as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
            extractFromPDF = ""
        return finalText

    def toCSV(self, goals, index, dict):
        filename = index + ".csv"
        direc = 'Term/' + goals + "/"
        with open(direc + filename, 'w') as csvFile:
            w = csv.DictWriter(csvFile, dict.keys())
            w.writeheader()
            w.writerow(dict)

    def fromPDFFolders(self, goalName):
        count = 0
        directory = (glob.glob("../Data Set/" + goalName + "/*.pdf"))
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
                # print("Count PDF #: " + str(count))
        return count

    def preProcessing(self, text):
        text = preProc.removeSpecialCharacters(text)
        text = preProc.manual_tokenization(text)
        text = preProc.removeStopWords(text)

        # for t in text:
        #     finalText.append(preProc.removeNumbers(t))
        return text

    def populateClass(self, text):
        initialList = {}
        for t in text:
            initialList[t.lower()] = 0
        initialList = preProc.removeNumbers(initialList)
        initialList = preProc.cleanRawFrequency(initialList)
        return initialList

    def term_vectors(self, text, populatedDict):
        for word in text:
            for dic in populatedDict:
                if (dic == word):
                    populatedDict[dic] = populatedDict[dic] + 1
        return populatedDict

    def term_frequency(self, populatedDict):
        for dic in populatedDict:
            populatedDict[dic] = populatedDict[dic] / len(populatedDict)
        return populatedDict

    def inverse_frequency(self, features, dict1, dict2, dict3):
        idf = features
        number_of_documents = self.countNumberofDocs()
        number_of_word = 0
        for f in features:
            if dict1.__contains__(f):
                number_of_word += 1
            if dict2.__contains__(f):
                number_of_word += 1
            if dict3.__contains__(f):
                number_of_word += 1
            idf[f] = math.log(number_of_documents / number_of_word + 1)
            number_of_word = 0
        return idf

    def computeTF_IDF(self, tf, idf):
        tf_idf = tf
        for word in tf:
            if idf.__contains__(word):
                tf_idf[word] = float(tf_idf[word]) * float(idf[word])
            else:
                tf_idf[word] = 0

        return tf_idf

    def countNumberofDocs(self):
        count = 0
        dir_path = r'Term\Goal 2'
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count

    def mergeDictionaries(self, dict1, dict2, dict3):
        dict3 = {**dict1, **dict2}
        return dict3

    def csvToDict(self, number):
        diction = {}
        filename = "Term/Goal 1/" + str(number) + ".csv"
        # df = pd.read_csv(filename)
        # for index, rows in df.iterrows():
        #     d = rows.to_dict()
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                diction = line
        return diction

    def convertingToDP(self, featureSet, tf_idf):
        df = pd.DataFrame.from_dict(tf_idf)
        df2 = df.replace(np.nan, 0)
        df2.to_csv('file_name.csv')
        return df2


if __name__ == '__main__':
    goals = ['Goal 1', 'Goal 2']
    emptyDict = {}
    term_freq = {}
    finalFeatures = {}
    dict1 = {}
    dict2 = {}
    dict3 = {}
    idf = {}
    tf_idf = {}
    tf_idf2 = {}
    tf_idf3 = {}
    passToPD = [{}]
    processedText = ""
    processedText = Processing(processedText)
    TFIDF = Processing(processedText)
    # preprocessedText = TFIDF.getFromPDF()

    # print("Before Processing: " + str(len(preprocessedText)))
    # preprocessedText = TFIDF.preProcessing(preprocessedText)
    # print("After Processing: " + str(len(preprocessedText)))
    # emptyDict = TFIDF.populateClass(preprocessedText)
    # term_freq = TFIDF.term_vectors(preprocessedText, emptyDict)
    # term_freq = TFIDF.term_frequency(emptyDict)
    # TFIDF.toCSV("Goal 2", str(3), term_freq)
    # print("CSV File Created!")
    dict1 = TFIDF.csvToDict(1)
    dict2 = TFIDF.csvToDict(2)
    dict3 = TFIDF.csvToDict(3)

    finalFeatures = TFIDF.mergeDictionaries(dict1, dict2, dict3)
    # print("DICT 1: " + str(len(dict1)) + "DICT 2: " + str(len(dict2)))
    for v in finalFeatures:
        finalFeatures[v] = 0.0
    idf = TFIDF.inverse_frequency(finalFeatures, dict1, dict2, dict3)

    tf_idf = TFIDF.computeTF_IDF(dict1, idf)
    tf_idf2 = TFIDF.computeTF_IDF(dict2, idf)
    tf_idf3 = TFIDF.computeTF_IDF(dict3, idf)

    passToPD = tf_idf, tf_idf2, tf_idf3
    # tfidf.toCSV("Goal 1", "TFIDF", tf_idf)
    tf_idf = TFIDF.convertingToDP(finalFeatures, passToPD)
    print(tf_idf)
