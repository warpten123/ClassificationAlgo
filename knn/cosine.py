import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def preprocess_documents(docs):
    preprocessed_docs = []
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

        # Join tokens back into a string
        # preprocessed_doc = " ".join(lemmatized_tokens)
        # preprocessed_docs.append(preprocessed_doc)

    return lemmatized_tokens


def getUniqueWords(preProcessedDocs):
    unique = {}
    for str in preProcessedDocs:
        for str2 in str:
            unique[str2] = 0
    return unique


def getTerm(unique, length, listOfTokens=list,):
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


def getTermFreq(unique, length, listOfTokens=list,):

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


def inverse(unique, preProcessedDocs, tf=[{}]):

    num_of_docs = len(preProcessedDocs)
    idf = {}
    finalIDF = {}
    count = 0
    for str in unique:
        idf[str] = 0
    for str in idf:
        for str2 in tf:
            if str in str2:
                idf[str] = idf[str] + str2[str]

    for str in idf:
        finalIDF[str] = math.log10(num_of_docs / idf[str])
    return finalIDF


def calculateTFIDF(listofDict, idf, tf_idf):
    temp = {}
    count = 1
    for list in listofDict:
        temp = list
        for features in idf:
            if temp.__contains__(features):
                temp[features] = temp[features] * idf[features]
        tf_idf.append(temp)

    return tf_idf


def calculateNewData(unique, newData, idf):
    tf = [{}]
    tv = [{}]
    final = [{}]
    preProcessedDocs = preprocess_documents(newData)
    uniqueNew = getUniqueWords(preProcessedDocs)

    for listOfTokens in preProcessedDocs:
        tf.append(getTermFreq(uniqueNew, len(listOfTokens), listOfTokens))
        tv.append(getTerm(uniqueNew, len(listOfTokens), listOfTokens))
    tf.pop(0)
    tv.pop(0)

    # idf = inverse(uniqueNew, tv)
    # final = calculateTFIDF(tf, idf, final)
    # final.pop(0)
    # values = []
    # for doc in final:
    #     values.append(doc.values())
    # return values


def getTFIDF(documents):
    tv = [{}]
    tf = [{}]
    final = [{}]
    preProcessedDocs = preprocess_documents(documents)
    unique = getUniqueWords(preProcessedDocs)

    # print(unique)
    for listOfTokens in preProcessedDocs:
        tf.append(getTermFreq(unique, len(listOfTokens), listOfTokens))
        tv.append(getTerm(unique, len(listOfTokens), listOfTokens))
    tf.pop(0)
    tv.pop(0)
    idf = inverse(unique, preProcessedDocs, tv)
    final = calculateTFIDF(tf, idf, final)
    final.pop(0)
    values = []
    for doc in final:
        values.append(doc.values())

    # newVector = calculateNewData(unique, newData, final)
    return values


def getCosine(oldDoc, count):
    newVector = oldDoc[len(oldDoc)-1]
    del oldDoc[-1]
    goals = ["Goal 1: No Poverty", "Goal 2: Zero Hunger",
             "Goal 3: Good Health and Well-Being", "Goal 4: Quality Education", "Goal 5: Innovation"]

    for val in oldDoc:
        val2 = val
        t1 = []
        t2 = []
        for shit in newVector:
            t1.append(shit)
        for shit in val2:
            t2.append(shit)
        dotProduct = 0
        for i in range(len(t1)):
            dotProduct = dotProduct + (t1[i] * t2[i])
        magnitude = 0
        for i in range(len(t1)):
            magnitude = magnitude + (math.pow(t1[i], 2))
        print(round((dotProduct / magnitude) * 100, 2), "%")


documents = [
    "Poverty in the Philippines is so high. Family earns less than 10k a month contributing to the nation's poverty rate. It is by far the poorest country in the world.",
    "South Africans are dying because of hunger. Hungry children can be seen even on the streets of their capital city. Hunger is the number one killer.",
    "Mental health and well being are some times a taboo topic for tradiotional families. Mental health topics are often times discraded",
    "Quality education is for everyone. It is a basic human right. A child's education should be a top priority for parents",
    "Technology innovation is a must. Artificial Intelligence are in a boom right now especially with the introduction of CHATGPT",
]
newDocs = []
# newData = ["Artificial Intelligence like chatGPT is killing us."]

firstTimeRun = True
cont = True
count = 0
while (cont != False):
    newData = input("Enter Document: ")
    documents.append(newData)
    values = getTFIDF(documents)
    count += 1
    getCosine(values, count)
    del documents[-1]
