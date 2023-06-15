import csv
import glob
import math
import os
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns


class Cosine():

    def checkDataSet(self):
        cont = False
        csv = "TFIDF.csv"
        directory = (
            glob.glob("tfidf/Results/PreProcessed/" + "/*.txt"))
        for file in directory:
            if (file == "tfidf/Results/PreProcessed\checker.txt"):
                cont = True
        return cont

    def checkLastData(self):
        cont = False

        directory = (
            glob.glob(f"tfidf/Results/PreProcessed/" + "/*.txt"))
        for file in directory:
            if (file == "tfidf/Results/PreProcessed\PreProcessed 18.txt"):
                cont = True
        return cont

    def preprocess_documents(self, docs):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        for doc in docs:
            # Tokenize document
            tokens = word_tokenize(doc)
            # Remove stopwords and punctuations, and convert to lowercase
            filtered_tokens = [
                token.lower() for token in tokens if token.isalnum() and token not in stop_words]
            # Lemmatize tokens
            lemmatized_tokens.append([lemmatizer.lemmatize(
                token) for token in filtered_tokens])

        return lemmatized_tokens

    def getUniqueWords(self, preProcessedDocs, type):
        unique = {}
        for str in preProcessedDocs:
            for str2 in str:
                unique[str2] = 0
        return unique

    def getTerm(self, unique, length, listOfTokens=list,):
        term_vec = {}
        for str in unique:
            term_vec[str] = 0
        for token in term_vec:
            for str2 in listOfTokens:
                if (token == str2):
                    term_vec[token] = term_vec[token] + 1

        # for token in term_vec:
        #     term_vec[token] = term_vec[token] / length

        return term_vec

    def getTermFreq(self, unique, length, listOfTokens=list,):

        term_vec = {}
        for str1 in unique:
            term_vec[str1] = 0
        for token in term_vec:
            for str2 in listOfTokens:
                if (token == str2):
                    term_vec[token] = term_vec[token] + 1
        for token in term_vec:
            term_vec[token] = term_vec[token] / length

        return term_vec

    def inverse(self, unique, preProcessedDocs, tf=[{}]):
        num_of_docs = len(preProcessedDocs)
        idf, finalIDF = {}, {}
        for str in unique:
            idf[str] = 0
        for str in idf:
            for str2 in tf:
                if str in str2:
                    idf[str] = idf[str] + str2[str]
        for str in idf:
            finalIDF[str] = math.log10(num_of_docs / idf[str])
        return finalIDF

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

    def getTFIDF(self, documents, testing):
        tv, tf, final = [{}], [{}], [{}]
        index = 1

        if (self.checkDataSet() == False):
            preProcessedDocs = self.preprocess_documents(documents)
            unique = self.getUniqueWords(preProcessedDocs, False)

            for token in preProcessedDocs:
                self.writeListToTxt(' '.join(token), index)
                index += 1
            self.addChecker()
        else:
            preProcessedDocs = documents
            unique = self.getUniqueWords(preProcessedDocs, True)

        for listOfTokens in preProcessedDocs:
            tf.append(self.getTermFreq(
                unique, len(listOfTokens), listOfTokens))
            tv.append(self.getTerm(unique, len(listOfTokens), listOfTokens))
        tf.pop(0)
        tv.pop(0)
        idf = self.inverse(unique, preProcessedDocs, tv)
        final = self.calculateTFIDF(tf, idf, final)
        final.pop(0)
        values = []
        for doc in final:
            values.append(doc.values())
        self.test(values[0])

        if (testing == False):
            tf_idf = self.convertingToDP(final)
        return values

    def check_if_list(self, param):
        if isinstance(param, list):
            print("Parameter is a list")
        else:
            print("Parameter is not a list")

    def convertingToDP(self, tf_idf):
        df = pd.DataFrame.from_dict(tf_idf)
        df2 = df.replace(np.nan, 0)
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2.to_csv('tfidf/Results/TFIDF.csv')
        return df2

    def heatMap(self, tf_idf):
        df = pd.DataFrame.from_dict(tf_idf)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap='Blues', annot=True, fmt='.2f')
        plt.title('TF-IDF Visualization')
        plt.xlabel('Words')
        plt.ylabel('Documents')
        plt.show()
        # df2 = df.replace(np.nan, 0)
        # df2.to_csv('tfidf/Results/TFIDF.csv')
        # return df2

    def TFIDFForConfusion(self, documents, testing):
        tv, tf, final = [{}], [{}], [{}]
        index = 1
        preProcessedDocs = documents
        unique = self.getUniqueWords(preProcessedDocs, True)

        for listOfTokens in preProcessedDocs:
            tf.append(self.getTermFreq(
                unique, len(listOfTokens), listOfTokens))
            tv.append(self.getTerm(unique, len(listOfTokens), listOfTokens))
        tf.pop(0)
        tv.pop(0)
        idf = self.inverse(unique, preProcessedDocs, tv)
        final = self.calculateTFIDF(tf, idf, final)
        final.pop(0)
        return final

    def getCosine(self, oldDoc, count):
        newVector, cosine = oldDoc[len(oldDoc)-1], []
        counter = 0
        classifier = {}
        del oldDoc[-1]
        goals = ["Goal 1: No Poverty", "Goal 2: Zero Hunger", "Goal 3: Good Health and Well-Being", "Goal 4: Quality Education", "Goal 5: Gender Equality", "Goal 6: Clean Water and Sanitation", "Goal 7: Affordable and Clean Energy", "Goal 8: Decent Work and Economic Growth",
                 "Goal 9: Industry, Innovation, and Infrastructure", "Goal 10: Reduced Inequalities", "Goal 11: Sustainable Cities and Communities", "Goal 12: Responsible Consumption and Production", "Goal 13: Climate Action", "Goal 14: Life Below Water", "Goal 15: Life on Land", "Goal 16: Peace, Justice and Strong Institutions", "Goal 17: Partnership for the Goals"
                 ]
        for val in oldDoc:
            val2 = val
            vector1, vector2 = [], []
            dotProduct, magnitude, magnitude1, magnitude2 = 0, 0, 0, 0
            for newvec in newVector:
                vector1.append(newvec)
            for oldVar in val2:
                vector2.append(oldVar)
            dotProduct = np.dot(vector1, vector2)
            magnitude1 = math.sqrt(
                sum(component ** 2 for component in newVector))
            magnitude2 = math.sqrt(sum(component ** 2 for component in val2))
            magnitude = magnitude1 * magnitude2
            cosine.append(round(dotProduct/magnitude, 5))
            percent = round(
                (dotProduct / magnitude) * 100, 2)
            classifier[goals[counter]] = percent
            counter += 1

        sorted_dict = dict(
            sorted(classifier.items(), key=lambda item: item[1], reverse=True))

        return sorted_dict

    def get_cosine_matrix(self, oldDoc):
        del oldDoc[-1]
        test = []
        count = 0
        dotProduct = 0
        magnitude = 0
        for i in range(len(oldDoc)):
            for j in range((len(oldDoc))):
                dotProduct = dotProduct + (oldDoc[i] * oldDoc[j])
                magnitude = magnitude + (math.pow(oldDoc[i], 2))
                test.append(round(dotProduct/magnitude, 2))

        return str(count)

    def csvToDict(self):  # not used
        with open('tfidf/Results/TFIDF.csv', encoding="utf8") as f:
            a = [{k: float(v) for k, v in row.items()}
                 for row in csv.DictReader(f, skipinitialspace=True)]
        print("FROM CSV GOAL 1: ", a[0])
        return a

    def extractAllPDF(self, goal):
        directory = (glob.glob("tfidf/Data Set/" + goal + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extractedText = page.extract_text()
                    finalText = finalText + extractedText

        return finalText

    def writeListToTxt(self, training, index):
        with open(r"tfidf/Results/PreProcessed/PreProcessed " + str(index) + ".txt", 'w', encoding="utf8") as fp:
            fp.write(training)

    def addChecker(self):
        with open(r"tfidf/Results/PreProcessed/checker.txt", 'w', encoding="utf8") as fp:
            fp.write("checker")

    def readListFromTxt(self, index):
        string = ""
        with open(r"tfidf/Results/PreProcessed/PreProcessed " + str(index) + ".txt", 'r', encoding="utf8") as f:
            for line in f:
                string = string + line.strip()

                # add current item to the list

        return string

    def removeNewData(self):
        return os.remove("tfidf/Results/PreProcessed/" + "PreProcessed 18.txt")

    def storeTraining(self, preProcessedDocs):
        index = 1
        for token in preProcessedDocs:
            str1 = ' '.join(token)
            self.writeListToTxt(str1, index)
            index += 1
            str1 = " "

    def extractTraining(self):
        index = 17
        string = ""
        extractedTraining = []
        for i in range(index):
            with open(r"tfidf/Results/PreProcessed/PreProcessed " + str(i+1) + ".txt", 'r', encoding="utf8") as f:
                for line in f:
                    string = line.split()
                    extractedTraining.append(string)
                    string = ""
        return extractedTraining

    def trainingPhase(self):
        print("Starting Training")
        trainingDocs = []
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
                 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
                 'Goal 13',
                 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
                 ]
        for goal in goals:
            trainingData = self.extractAllPDF(goal)
            trainingDocs.append(trainingData)
        self.getTFIDF(trainingDocs, False)
        print("End Training")

    def test(self, training):
        with open(r"tfidf/Results/" + "test.txt", 'w', encoding="utf8") as fp:
            fp.write(str(training))

    def classifyResearch(self, data, testing):
        start_time = time.time()
        count = 0
        trainingDocs, newDocs = [], []
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
                 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
                 'Goal 13',
                 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
                 ]
        if (self.checkDataSet() == False):
            for goal in goals:
                trainingData = self.extractAllPDF(goal)
                trainingDocs.append(trainingData)
        else:
            trainingDocs = self.extractTraining()
            newDocs.append(data)
            newData = self.preprocess_documents(newDocs)
            data = newData[0]

        trainingDocs.append(data)
        values = self.getTFIDF(trainingDocs, testing)
        count += 1
        if (self.checkLastData()):
            self.removeNewData()
        result = self.getCosine(values, count)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Classify Time: ", execution_time, " seconds")
        return result
        # a = self.csvToDict()
        # values = []
        # del a[-1]
        # for i in range(len(a)):
        #     values.append(a[i].values())
        # self.getCosine(values, 0)
        # return "test"


##
# dic = {}
#   vector1 = [1, 4, 5, 6, 7, 5]
#         vector2 = [7, 6, 5, 4, 2, 8]
#         dotProduct = 0
#         magnitude = 0
#         dotProduct = np.dot(vector1, vector2)
#         dic = {}
#         magnitude1 = math.sqrt(sum(component ** 2 for component in vector2))
#         magnitude2 = math.sqrt(sum(component ** 2 for component in vector1))
#         magnitude = magnitude1 * magnitude2
#         print(magnitude)
#         dic['shit'] = dotProduct / magnitude
#         return dic
#
#
# ###
