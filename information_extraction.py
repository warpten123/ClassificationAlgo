import nltk

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nameparser.parser import HumanName
from nltk.corpus import wordnet
import pdfplumber
from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger('stanford-ner/all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner.jar')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text = '''
This is a sample text that contains the name Alex Smith who is one of the developers of this project.
You can also find the surname Jones here.
'''
person_list = []
person_names = person_list


class Extract():
    def __init__(self):
        self = self

    def extractNames(self, text):
        nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
        for nltk_result in nltk_results:
            if type(nltk_result) == Tree:
                name = ''
                for nltk_result_leaf in nltk_result.leaves():
                    name += nltk_result_leaf[0] + ' '
                print('Type: ', nltk_result.label(), 'Name: ', name)

    def extractFromPDF(self):
        extractedText = " "
        with pdfplumber.open("../EUL_ A Digital Research Repository System.pdf") as pdf:
            first_page = pdf.pages[0]
            extractedText = first_page.extract_text()
        return extractedText

    def get_human_names(self, text):
        for sent in nltk.sent_tokenize(text):
            tokens = nltk.tokenize.word_tokenize(sent)
            tags = st.tag(tokens)
            for tag in tags:
                if tag[1] == 'PERSON':
                    print(tag)


text = """
"EUL: A Digital Research Repository System A Thesis Presented to
The Faculty of the School of Computer Studies
Department of the 
University of San Jose-Recoletos
Cebu City, Philippines
In Partial Fulfillment
Of the Requirements for Thesis 1
Members
Cristopher Bohol
Paul Joshua Premacio 
Thesis Adviser
Dr. Lorna Miro
Date 2022
"
"""
extract = Extract()
extractedText = extract.extractFromPDF()
# extract.extractNames(extractedText)
print(extractedText)
names = extract.get_human_names(extractedText)
for person in person_list:
    person_split = person.split(" ")
    for name in person_split:
        if wordnet.synsets(name):
            if (name in person):
                person_names.remove(person)
                break
print(person_names)
