
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
from fpdf import FPDF
from collections import ChainMap
# use this when running main.py in backend
from tfidf.text_processing import PreProcessing
# from text_processing as preProc uncomment this shit if you want to run this file only
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


class Processing():
    def __init__(self, path):
        self.path = path

    def getFromPDF(self, filename):  # notused
        finalText = " "
        with pdfplumber.open('assets/upload/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
            extractFromPDF = ""
        return finalText

    def toCSV(self, goals, index, dict):  # noteused
        filename = str(index) + ".csv"
        direc = 'tfidf/Term/' + goals + "/"
        with open(direc + filename, 'w') as csvFile:
            for list in dict:
                w = csv.DictWriter(csvFile, list.keys())
                w.writeheader()
                w.writerow(list)

    def fromPDFFolders(self, goalName):
        tf_idf = {}
        count = 0
        directory = (glob.glob("../Data Set/" + goalName + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
                print("Count PDF #: " + str(count))
                for page in pdf.pages:
                    extractedText = page.extract_text()
            finalText = finalText + extractedText
            print("(Extracted Text): " +
                  str(count) + str(len(finalText)))
            extractedText = ""
            tf_idf = self.mainProcessing(finalText, goal, count)
        return tf_idf

    def preProcessing(self, text):
        preProc = PreProcessing()
        text = preProc.removeSpecialCharacters(text)
        text = preProc.manual_tokenization(text)
        text = preProc.removeStopWords(text)
        text = preProc.toLowerCase(text)
        return text

    def populateClass(self, text):
        preProc = PreProcessing()
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

    def inverse_frequency(self, features,  listOfDict=[{}]):
        idf = features
        number_of_documents = 17
        number_of_word = 0
        count = 0
        for col in features:
            for f in listOfDict:
                if f.__contains__(col):
                    number_of_word += 1
            idf[col] = math.log10(number_of_documents / number_of_word)
            number_of_word = 0
        return idf

    def calculateTFIDF(self, listofDict, idf, tf_idf):
        temp = {}
        count = 1
        for list in listofDict:
            temp = list
            for features in idf:
                if temp.__contains__(features):
                    temp[features] = temp[features] * idf[features]
            tf_idf.append(temp)

        return tf_idf
        # return tf_idf

    def computeTF_IDF(self, tf, idf):
        tf_idf = tf
        for word in tf:
            if idf.__contains__(word):
                tf_idf[word] = float(tf_idf[word]) * float(idf[word])
            else:
                tf_idf[word] = 0

        return tf_idf

    def countNumberofDocs(self):  # noteused
        count = 0
        dir_path = r"Term/Goal 1"
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count

    def mergeDictionaries(self, listofDict):
        finalDict = {**listofDict}
        return finalDict

    def csvToDict(self, number):  # not used
        diction = {}
        filename = "Term/Goal 2/" + str(number) + ".csv"
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                diction = line
        return diction

    def convertingToDP(self, featureSet, tf_idf):
        df = pd.DataFrame.from_dict(tf_idf)
        df2 = df.replace(np.nan, 0)
        df2.to_csv('tfidf/Results/TFIDF.csv')
        return df2

    def extractAllPDF(self, goal):
        count = 0
        directory = (glob.glob("tfidf/Data Set/" + goal + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
                print(goal + " PDF #: " + str(count))
                for page in pdf.pages:
                    extractedText = page.extract_text()
                    print("Words Length: " + str(len(extractedText)))
                    finalText = finalText + extractedText
        return finalText

    def listToPDF(self, processedText, goal):
        with open(r"tfidf/Text Files/" + goal + ".txt", 'w') as fp:
            fp.write(' '.join(processedText))
        f = open("tfidf/Text Files/" + goal + ".txt", "r")
        pdf = FPDF()
        pdf.set_font('Times', '', 12)
        pdf.add_page()
        for x in f:
            pdf.cell(200, 10, txt=x, ln=1, align='C')
        pdf.output("tfidf/Training Set/" + goal + " Training Set" + ".pdf")

    def mergeAllDict(self, l):
        d = {}
        for dictionary in l:
            d.update(dictionary)
        return d

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        temp = []
        for str in text:
            lemma = lemmatizer.lemmatize(str)
            temp.append(lemma)
        return temp

    def createTFIDF(self, rawText):
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
                 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
                 'Goal 13',
                 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
                 ]
        TFIDF = Processing(rawText)
        tf = [{}]  # create list of dicts
        count = 1
        final_features = {}
        idf = {}
        tf_idf = [{}]
        temp = {}
        merge = {}
        for goal in goals:
            rawText = TFIDF.extractAllPDF(goal)
            preprocessedText = TFIDF.preProcessing(rawText)
            preprocessedText = TFIDF.lemmatization(preprocessedText)
            TFIDF.listToPDF(preprocessedText, goal)
            temp = TFIDF.populateClass(preprocessedText)
            temp = TFIDF.term_vectors(preprocessedText, temp)
            temp = TFIDF.term_frequency(temp)
            tf.append(temp)
            count += 1
        merge = TFIDF.mergeAllDict(tf)
        idf = TFIDF.inverse_frequency(merge, tf)
        tf_idf = TFIDF.calculateTFIDF(tf, idf, tf_idf)
        tf_idf = TFIDF.convertingToDP(merge, tf_idf)
        return tf_idf

    def insertNewData(self, path):
        # holy shit
        length = 0
        rawText = self.getFromPDF(path)
        preprocessedText = self.preProcessing(rawText)
        lemmatized = self.lemmatization(preprocessedText)
        print(lemmatized)
        return lemmatized


if __name__ == '__main__':
    rawText = ""
    TFIDF = Processing(rawText)
    TFIDF.createTFIDF(rawText)
