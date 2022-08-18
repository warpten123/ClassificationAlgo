from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files
import re

texts_from_file = docx2txt.process("Introduction.docx")
# print(texts_from_file)

def manual_tokenization(text):
    container = ""
    newText = []
    for i in range(len(text)):
        if(text[i] != ' ' and text[i] != '\t'):
            container = container + text[i]
            if(i == len(text)-1):
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

text,processedText = [],[]
text = texts_from_file

print("Initial Count of Words in File: " + str(len(text)))
processedText = removeSpecialCharacters(text)
processedText = toLowerCase(processedText)
processedText = manual_tokenization(processedText)
processedText = removeStopWords(processedText)
print("After Preprocessing: " + str(len(processedText)))
# processedText = text_tokenization(processedText)





# lowercase the content from text







# check for repeating words in text and save it as a map<word,count>
# text = dict([(word, text.count(word)) for word in set(text)])
# print(text)
# print(str(len(text)))
    





