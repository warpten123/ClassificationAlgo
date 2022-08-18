import docx2txt #lib for reading docx files

texts_from_file = docx2txt.process("Introduction.docx")
print(texts_from_file)


text = []
text = texts_from_file
print("Initial Count of Words in File: " + str(len(text)))

