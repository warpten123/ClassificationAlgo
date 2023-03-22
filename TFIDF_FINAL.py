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
from fpdf import FPDF
from collections import ChainMap
import nltk
# class word:
#     def __init__(self, word, count):
#         self.word = word
#         self.count = count

nltk.download('wordnet')
nltk.download('omw-1.4')


class Processing():
    def __init__(self, path):
        self.path = path

    def getFromPDF(self):
        finalText = " "
        # with pdfplumber.open(r'../Data Set/Goal 1/goal 1_3.pdf') as pdf:
        with pdfplumber.open(r'Goal 1 Training Set.pdf') as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
            extractFromPDF = ""
        return finalText

    def toCSV(self, goals, index, dict):
        filename = str(index) + ".csv"
        direc = 'Term/' + goals + "/"
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
        text = preProc.removeSpecialCharacters(text)
        text = preProc.manual_tokenization(text)
        text = preProc.removeStopWords(text)
        # text = preProc.removeNumbers(text)
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
        # for features in idf:
        #     for list in listofDict:
        #         temp = list
        #         print(temp)
        #         if list.__contains__(features):
        #             temp[features] = temp[features] * idf[features]
        #         else:
        #             temp[features] = 0.0
        # tf_idf.append(temp)

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

    def countNumberofDocs(self):
        count = 0
        dir_path = r"Term/Goal 1"
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count

    def mergeDictionaries(self, listofDict):
        finalDict = {**listofDict}
        return finalDict

    def csvToDict(self, number):
        diction = {}
        filename = "Term/Goal 2/" + str(number) + ".csv"
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
        df2.to_csv('TFIDF.csv')
        return df2

    def mainProcessing(self, preProcesssedText, goal, index):
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
        preprocessedText = TFIDF.preProcessing(processedText)
        emptyDict = TFIDF.populateClass(preprocessedText)
        term_freq = TFIDF.term_vectors(preprocessedText, emptyDict)
        term_freq = TFIDF.term_frequency(term_freq)
        TFIDF.toCSV(goal, str(index), term_freq)
        dict1 = TFIDF.csvToDict(index)
        # if index == 3:
    ##### NEW TFIDF###

    def extractAllPDF(self, goal):
        count = 0
        directory = (glob.glob("../Data Set/" + goal + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
                print("Count PDF #: " + str(count))
                for page in pdf.pages:
                    extractedText = page.extract_text()
                    print("length: " + str(len(extractedText)))
                    finalText = finalText + extractedText
            # print("(Extracted Text): " +
            #       str(count) + str(len(finalText)))

        # tf_idf = self.mainProcessing(finalText, goal, count)
        return finalText

    def listToPDF(self, processedText, goal):
        with open(r"Text Files/" + goal + ".txt", 'w') as fp:
            fp.write(' '.join(processedText))
        f = open("Text Files/" + goal + ".txt", "r")
        pdf = FPDF()
        pdf.set_font('Times', '', 12)
        pdf.add_page()
        for x in f:
            pdf.cell(200, 10, txt=x, ln=1, align='C')
        pdf.output("Training Set/" + goal + " Training Set" + ".pdf")

    def mergeAllDict(self, l):
        d = {}
        for dictionary in l:
            d.update(dictionary)
        return d

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()

        return lemmatizer.lemmatize(text)

    def createTFIDF():
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
             'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
             'Goal 13',
             'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
             ]
        rawText = ""
        TFIDF = Processing(rawText)
        tf = [{}]  # create list of dicts
        count = 1
        final_features = {}
        idf = {}
        tf_idf = [{}]
        temp = {}
        merge = {}
        print(len(tf))
        # preprocessedText = TFIDF.getFromPDF()
        for goal in goals:
            rawText = TFIDF.extractAllPDF(goal)
            preprocessedText = TFIDF.lemmatization(rawText)
            preprocessedText = TFIDF.preProcessing(preprocessedText)
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
        ### END NEW TF IDF###
        
# if __name__ == '__main__':

    # TFIDF = Processing()
    # TFIDF.createTFIDF()
    # print(len(finalTF_IDF))
    # print(tf)
    # tf.append(TFIDF.term_frequency(tf[count]))
    # final_features = TFIDF.mergeAllDict(tf[count])
    # with open('your_file.txt', 'w') as f:
    #     for line in tf:
    #         for key in line:
    #             f.write("%s\n" % key)
    # for v in final_features:
    #     final_features[v] = 0.0
    # idf = TFIDF.inverse_frequency(final_features, tf)
    # print(tf[2])

    # print(final_features)
    # final_features = TFIDF.mergeAllDict(tf)
    # for v in final_features:
    #     final_features[v] = 0.0
    # print(len(final_features))
    # idf = TFIDF.inverse_frequency(final_features, tf)
    # print(idf)
    # print(final_features)
    # print(preprocessedText)
    # test = TFIDF.getFromPDF()
    # test = list(test.split(" "))
    # print(test)

    # test = TFIDF.getFromPDF()
    # print("FROM TEST: " + str(len(test)))

    # print(len(processedText))
    # print("Before Processing: " + str(len(preprocessedText)))
    # preprocessedText = TFIDF.preProcessing(preprocessedText)
    # print("After Processing: " + str(len(preprocessedText)))
    # emptyDict = TFIDF.populateClass(preprocessedText)
    # term_freq = TFIDF.term_vectors(preprocessedText, emptyDict)
    # term_freq = TFIDF.term_frequency(emptyDict)
    # TFIDF.toCSV("Goal 2", str(3), term_freq)
    # print("CSV File Created!")

    # dict1 = TFIDF.csvToDict(1)
    # dict2 = TFIDF.csvToDict(2)
    # dict3 = TFIDF.csvToDict(3)

    # finalFeatures = TFIDF.mergeDictionaries(dict1, dict2, dict3)
    # # print("DICT 1: " + str(len(dict1)) + "DICT 2: " + str(len(dict2)))
    # for v in finalFeatures:
    #     finalFeatures[v] = 0.0
    # idf = TFIDF.inverse_frequency(finalFeatures, dict1, dict2, dict3)

    # tf_idf = TFIDF.computeTF_IDF(dict1, idf)
    # tf_idf2 = TFIDF.computeTF_IDF(dict2, idf)
    # tf_idf3 = TFIDF.computeTF_IDF(dict3, idf)

    # passToPD = tf_idf, tf_idf2, tf_idf3
    # # tfidf.toCSV("Goal 1", "TFIDF", tf_idf)
    # tf_idf = TFIDF.convertingToDP(finalFeatures, passToPD)
    # print(tf_idf)
