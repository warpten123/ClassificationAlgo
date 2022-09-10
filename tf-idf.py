from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files
import re
import pandas as pd
import numpy as np


texts_from_file = docx2txt.process("Introduction.docx")
print(texts_from_file[0])


# sentence tokenization
def sentence_tokenization(texts_from_file):
    # manual tokenization by sentences
    # if word ends with . or ! or ?  then it is a sentence
    sentences = []
    container = ""
    for i in range(len(texts_from_file)):
        if(texts_from_file[i] != '.' and texts_from_file[i] != '!' and texts_from_file[i] != '?'
        and texts_from_file[i] != '\n'and texts_from_file[i] != '\xa0'and texts_from_file[i] != '\t'):
            container = container + texts_from_file[i]
            if(i == len(texts_from_file)-1):
                sentences.append(container)
        else:
            sentences.append(container)
            container = ""
    return sentences


sentences = sentence_tokenization(texts_from_file)
# print(sentences)

# word tokenization
def word_tokenization(sentences):
    words = []
    for i in range(len(sentences)):
        words.append(sentences[i].split())
    return words

word = word_tokenization(sentences)
# print(word)

# remove empty array in word
def removeEmptyArray(word):
    word = [x for x in word if x != []]
    return word

word = removeEmptyArray(word)
# print(word)

# remove the value in word from index 0 - 3
def removeValue(word):
    # remove index 0 to 2
    for i in range(3):
        word.pop(0)
    return word

word = removeValue(word)
print(word)


def toLowerCase(text):
    print(text)
    # convert into lowercase

    
def removeStopWords(text):
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()
    # if text is in stop_words, remove it 
    text = [x for x in text if x not in stop_words]
    return text
    
def removeSpecialCharacters(text):
    # remove all special characters except / and -
    sym = re.sub(r'[^\w\s/-]', '', text)
    return sym

def save_to_file(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def pre_process(word):
    # loop each array in word
    for i in range(len(word)):
        # lowercase all elements under each array
        word[i] = [x.lower() for x in word[i]]
        # remove all stopwords under each_array
        word[i] = removeStopWords(word[i])
        # loop each element in word
        for j in range(len(word[i])):
            # if index value contains digit then remove it
            if word[i][j].isdigit():
                word[i][j] = ""
            # remove all special characters
            word[i][j] = removeSpecialCharacters(word[i][j])
            # check if first index of the current index is digit then remove it
            if word[i][j][0].isdigit():
                word[i][j] = ''
        # check if current index has empty string then remove it
        word[i] = [x for x in word[i] if x != '']
    return word

pre_process = pre_process(word)
# print(pre_process)
# use set to create a set of unique values
def create_set(pre_process):
    word_set = set()
    for i in range(len(pre_process)):
        for j in range(len(pre_process[i])):
            word_set.add(pre_process[i][j])
    return word_set

word_set = create_set(pre_process)
print(word_set)


index_dict = {}
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1


def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count

word_count = count_dict(pre_process)
print(word_count)

#Term Frequency
def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N

total_documents = len(pre_process)
#Inverse Document Frequency
 
def inverse_doc_freq(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_documents/word_occurance)

#Inverse Document Frequency
 
def inverse_doc_freq(word):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(total_documents/word_occurance)

def tf_idf(sentence):
    tf_idf_vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = termfreq(sentence,word)
        idf = inverse_doc_freq(word)
         
        value = tf*idf
        tf_idf_vec[index_dict[word]] = value 
    return tf_idf_vec

#TF-IDF Encoded text corpus
vectors = []
for sent in pre_process:
    vec = tf_idf(sent)
    vectors.append(vec)
    print(vec, sent)
 
print(vectors)
