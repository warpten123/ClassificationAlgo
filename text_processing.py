from concurrent.futures import process
from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files

texts_from_file = docx2txt.process("Introduction.docx")
print(texts_from_file)


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

text,processedText = [],[]
text = texts_from_file


print("Initial Count of Words in File: " + str(len(text)))

processedText = text_tokenization(text)
processedText = toLowerCase(processedText)
processedText = removeStopWords(processedText)
print(processedText)
# tokenize texts_from_file


# lowercase the content from text







# check for repeating words in text and save it as a map<word,count>
# text = dict([(word, text.count(word)) for word in set(text)])
# print(text)
# print(str(len(text)))
    





