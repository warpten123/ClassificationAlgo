from multiprocessing.resource_sharer import stop
import docx2txt #lib for reading docx files

texts_from_file = docx2txt.process("Introduction.docx")
print(texts_from_file)


text = []
text = texts_from_file
print("Initial Count of Words in File: " + str(len(text)))

# tokenize texts_from_file
text = texts_from_file.split()
print(text)
print("Count of Words in File: " + str(len(text)))

# lowercase the content from text
text = [word.lower() for word in text]
print(text)

print(str(len(text)))


# list of stop words
# read the file in stop_words
with open('stopwords.txt', 'r') as f:
    stop_words = f.read().splitlines()
print(stop_words)
print(str(len(stop_words)))

# remove stop words from text
text = [word for word in text if word not in stop_words]
print(text)
print(str(len(text)))

# check for repeating words in text and save it as a map<word,count>
text = dict([(word, text.count(word)) for word in set(text)])
print(text)
print(str(len(text)))
    





