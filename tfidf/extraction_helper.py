import pdfplumber
import os
from tfidf.text_processing import PreProcessing
from nltk import pos_tag
from nltk.tokenize import word_tokenize


class Helper:
    def getRequiredChapters(self):  # get abstract, intro, res methodology
        return True

    def main_logic(self, filename):
        print("FILE: " + str(filename))
        abstract = self.getFromPDF(filename)
        return abstract
        # return self.getAbstract(rawText)

    def getFromPDF(self, filename):
        preProc = PreProcessing()
        count = 1
        finalText = " "
        final_abstract = " "
        with pdfplumber.open('assets/upload/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                # processedText = preProc.manual_tokenization(finalText)
                print("Count: " + str(count))
                check = self.getAbstract(finalText, count)
                if (check):
                    final_abstract = finalText
                    final_abstract = self.cleanString(final_abstract)
                    break
                count += 1
                final_abstract = " "
                finalText = " "
        return final_abstract

    def getAbstract(self, processedText, page):
        count = 0
        pageAbstract = 0
        abstract = False
        if (("ABSTRACT" in processedText or "Abstract" in processedText) and "TABLE OF CONTENTS" not in processedText):
            if (count == 0):
                abstract = True
                pageAbstract = page
            count += 1
            print("Abstract is in page: " + str(pageAbstract))
        return abstract

    def cleanString(self, text):
        if ("\n" in text):
            text = text.replace('\n', ' ')
        return text

    def getRules(self):
        rules = []
        file = open('tfidf/Results/rules.txt', 'r')
        Lines = file.readlines()
        for line in Lines:
            rules.append(line.strip())
        print(rules)
        return rules

    def populateRules(self):
        finalText = " "
        with pdfplumber.open('assets/' + "rules_data_set.pdf") as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
        finalText = self.cleanString(finalText)
        sentences = self.extract_sentences(finalText)
        pos_tagger = self.pos_tagging(sentences)
        return pos_tagger

    def extract_sentences(self, text):
        preProc = PreProcessing()
        sentences = preProc.dot_tokenization(text)
        return sentences

    def pos_tagging(self, sentences):
        list_of_rules = []
        for str in sentences:
            sentences_tag = pos_tag(word_tokenize(str))
            list_of_rules.append(sentences_tag)
        print(list_of_rules)
        tags = [[tag for word, tag in sent] for sent in list_of_rules]
        return tags

    # def getIntroduction(self,processedText,page):
