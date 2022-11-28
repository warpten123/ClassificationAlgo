from concurrent.futures import process
from hashlib import new
from pathlib import Path
from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import k_nearest_neighbor as knn
import csv
import glob
import os
class TFIDF():
    #global array
    arr = [{
        "index": 0,
        "word": "",
        "value": 0
    }]
    ###
    def __init__(self, path):
        self.path = path

    def print(self):
        print(self.path)

    # sentence tokenization
    def sentence_tokenization(self):
        # manual tokenization by sentences
        # if word ends with . or ! or ?  then it is a sentence
        sentences = []
        container = ""
        for i in range(len(self.path)):
            if(self.path[i] != '.' and self.path[i] != '!' and self.path[i] != '?'
            and self.path[i] != '\n'and self.path[i] != '\xa0'and self.path[i] != '\t'):
                container = container + self.path[i]
                if(i == len(self.path)-1):
                    sentences.append(container)
            else:
                sentences.append(container)
                container = ""
        return sentences
    
    # word tokenization
    def word_tokenization(self, sentences):
        words = []
        for i in range(len(sentences)):
            words.append(sentences[i].split())
        return words

    # remove empty array in word
    def removeEmptyArray(self, word):
        word = [x for x in word if x != []]
        return word

    # remove the value in word from index 0 - 3
    def removeValue(self, word):
        # remove index 0 to 2
        for i in range(3):
            word.pop(0)
        return word

    # def toLowerCase(self, text):
    #     print(text)
    #     # convert into lowercase

    def removeStopWords(self, text):
        with open('stopwords.txt', 'r') as f:
            stop_words = f.read().splitlines()
        # if text is in stop_words, remove it 
        text = [x for x in text if x not in stop_words]
        return text
        
    def removeSpecialCharacters(self, text):
        # remove all special characters except / and -
        sym = re.sub(r'[^\w\s/-]', '', text)
        return sym

    def save_to_file(self, filename, lines):
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    def pre_process(self, word):
        # loop each array in word
        for i in range(len(word)):
            # lowercase all elements under each array
            word[i] = [x.lower() for x in word[i]]
            # remove all stopwords under each_array
            word[i] = self.removeStopWords(word[i])
            # loop each element in word
            for j in range(len(word[i])):
                # if index value contains digit then remove it
                if word[i][j].isdigit():
                    word[i][j] = ""
                # remove all special characters
                word[i][j] = self.removeSpecialCharacters(word[i][j])
                # check if first index of the current index is digit then remove it
                if word[i][j].isdigit():
                    word[i][j] = ''
            # check if current index has empty string then remove it
            word[i] = [x for x in word[i] if x != '']
        return word

    # use set to create a set of unique values
    def create_set(self, pre_process):
        word_set = set()
        for i in range(len(pre_process)):
            for j in range(len(pre_process[i])):
                word_set.add(pre_process[i][j])
        return word_set

    def count_dict(self, sentences):
        word_count = {}
        for word in word_set:
            word_count[word] = 0
            for sent in sentences:
                if word in sent:
                    word_count[word] += 1
        return word_count

    # Term Frequency
    def termfreq(self, document, word):
        N = len(document)
        occurance = len([token for token in document if token == word])
        return occurance/N

    # Inverse Document Frequency
    def inverse_doc_freq(self, word, total_documents):
        try:
            word_occurance = word_count[word] + 1
        except:
            word_occurance = 1
        return np.log(total_documents/word_occurance)

    def tf_idf(self, sentence, word_set, index_dict):
        for word in sentence:
            tf = self.termfreq(sentence, word)
            idf = self.inverse_doc_freq(word, len(sentence))
            value = tf*idf
            self.arr.append({
                "index": index_dict[word],
                "word": word,
                "value": value
            })

    #TF-IDF Encoded text corpus
    def encoded_corpus(self, pre_process, word_set, index_dict):
        vectors = []
        for sent in pre_process:
            vec = self.tf_idf(sent, word_set, index_dict)
            vectors.append(vec)
            # print(vec, sent)
        return vectors

    def plot_vectors(self, vectors):
        # plot the vectors with labels
        for i in range(len(vectors)):
            plt.plot(vectors[i], label = "sentence" + str(i))
        plt.legend()
        plt.show()
        
    def PDFProcessing(self,goalName):
        count = 0 
        directory = (glob.glob("Data Set/" + goalName + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1 
                print("Count PDF #: " + str(count))
                for page in pdf.pages:
                    extractedText =  page.extract_text()
            finalText = finalText + extractedText
            extractedText = ""
        return finalText
    
    def toCSV(self,fileName):
        direc = 'Data Set/' + fileName + "/"
        name = fileName + ".csv"
        with open(direc + name, 'w+',encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames = ['index', 'word', 'value'])
            writer.writeheader()
            writer.writerows(self.arr)
            f.close

# main program
goals = [
        "Goal 17"
]
if __name__=='__main__':
    # with pdfplumber.open(r'Goal 2 - Supporting Texts.pdf') as pdf:
    #     for page in pdf.pages:
    #         extractFromPDF = page.extract_text()
    # print(len(extractFromPDF))
    for goal in goals:
        print("Current " + goal)
        extractFromPDF = ""
        tfidf = TFIDF(extractFromPDF)
        extractFromPDF = tfidf.PDFProcessing(goal)
        tfidf = TFIDF(extractFromPDF)
        print(len(extractFromPDF))
        # tfidf.print()
        sentences = tfidf.sentence_tokenization()
        # print(sentences)
        word = tfidf.word_tokenization(sentences)
        # print(word)
        word = tfidf.removeEmptyArray(word)
        # print(word)    
        word = tfidf.removeValue(word)
        # print(word)
        pre_process = tfidf.pre_process(word)
        # print(len(pre_process))
        # print(pre_process)
        word_set = tfidf.create_set(pre_process)
        # print(word_set)
        word_count = tfidf.count_dict(pre_process)
        # print(word_count)
        total_documents = len(pre_process)
        # print(total_documents)
        index_dict = {}
        i = 0
        for word in word_set:
            index_dict[word] = i
            i += 1
        tfidf.encoded_corpus(pre_process, word_set, index_dict)
        # for i in range(len(tfidf.arr)):
        #     print(tfidf.arr[i])
        tfidf.toCSV(goal)
   
    

