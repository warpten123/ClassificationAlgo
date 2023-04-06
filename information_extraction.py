import nltk
import spacy
import text_processing as preProcess
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nameparser.parser import HumanName
from nltk.corpus import wordnet
import pdfplumber
from nltk.tag import StanfordPOSTagger

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
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
        with pdfplumber.open("EUL_ A Digital Research Repository System.pdf") as pdf:
            first_page = pdf.pages[0]
            extractedText = first_page.extract_text()
        return extractedText

    def posTagging(self, text):
        textList = preProcess.manual_tokenization(text)
        postTagged = nltk.pos_tag(textList)
        return postTagged

    def autoNER(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))
            
class mainExtract():
    def main():
        extract = Extract()
        extractedText = extract.extractFromPDF()
        print(extractedText)
        extract.autoNER(extractedText)
        postTagged = extract.posTagging(extractedText)
        extract.extractNames(extractedText)
        
mainExtract.main()
