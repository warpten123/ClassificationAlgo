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
        abstract = self.getFromPDFAbstract(filename)
        introduction = self.getFromPDFIntro(filename)
        method = self.getFromPDFMethod(filename)
        return {'abstract': abstract, 'introduction': introduction, 'method': method}
        # return self.getAbstract(rawText)

    def getFromPDFAbstract(self, filename):
        count = 1
        finalText = " "
        final_abstract = " "
        with pdfplumber.open('assets/upload/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                checkAbs = self.getAbstract(finalText, count)
                if (checkAbs):
                    final_abstract = finalText
                    final_abstract = self.cleanString(final_abstract)
                    break
                count += 1
                final_abstract = " "
                finalText = " "
        return final_abstract

    def getFromPDFIntro(self, filename):
        count = 1
        finalText = " "
        final_intro = " "
        with pdfplumber.open('assets/upload/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                checkAbs = self.getIntroduction(finalText)
                if (checkAbs):
                    final_intro = finalText
                    final_intro = self.cleanString(final_intro)
                    break
                count += 1
                final_intro = " "
                finalText = " "
        return final_intro

    def getFromPDFMethod(self, filename):
        count = 1
        finalText = " "
        final_method = " "
        with pdfplumber.open('assets/upload/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                checkAbs = self.getMethodology(finalText)
                if (checkAbs):
                    final_method = finalText
                    final_method = self.cleanString(final_method)
                    break
                count += 1
                final_method = " "
                finalText = " "
        return final_method

    def getAbstract(self, processedText, page):
        count = 0
        pageAbstract = 0
        abstract = False
        if (("ABSTRACT" in processedText or "Abstract" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                abstract = True
                pageAbstract = page
            count += 1
            print("Abstract is in page: " + str(pageAbstract))
        return abstract

    def getIntroduction(self, processedText):
        count = 0
        introduction = False
        if (("INTRODUCTION" in processedText or "Introduction" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                introduction = True
            count += 1
        return introduction

    def getMethodology(self, processedText):
        count = 0
        methodology = False
        if (("Research Methodology" in processedText or "RESEARCH METHODOLOGY" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                methodology = True
            count += 1
        return methodology

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
