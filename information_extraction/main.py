from collections import OrderedDict
import Python_Backend as backend
from information_extraction.document_extractor import DocumentExtractor
import re
import datetime
from dateutil import parser as date_parser
import random
import numpy
from nameparser import HumanName
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import pdfplumber
import glob
import os
import collections
from tfidf.text_processing import PreProcessing
from tfidf.TFIDF_FINAL import Processing
from nltk.tokenize import WhitespaceTokenizer
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
nltk.download('stopwords')


class InformationExtraction:
    def __init__(self, document_path):
        self.document_path = document_path
        self.nlp = spacy.load("en_core_web_sm")

    def extract_information(self, fromNode):
        print(self.document_path)
        extractor = DocumentExtractor(self.document_path)
        extracted_text = extractor.extract_text_from_document()
        # for i, extracted_text in enumerate(extracted_text, 1):
        #     print(f'Paragraph {i}: {extracted_text}')

        if extracted_text is not None:
            # information = self.process_extracted_text(extracted_text, fromNode)
            information = self.process_extracted_text(extracted_text, fromNode)
            return information
        else:
            print('Invalid file format. Please upload a PDF file.')

    def process_extracted_text(self, input_text, fromNode):
        information = {}
        information['title'] = self.extract_title(input_text)
        # Extract Department
        information['department'] = self.extract_department(input_text)
        # information['author'] = self.extract_person(input_text)
        # information['adviser'] = self.extract_adviser(input_text)
        information['authors'] = self.extract_names(input_text, fromNode)
        # Extract published date
        information['published_date'] = self.extract_published_date(input_text)
        # Return extracted information
        return information

    def extract_names(self, extract_text, fromNode):
        listOfLast = []
        listOfFirst = []
        count = 0
        for i in range(len(fromNode)):
            count += 1
            print(count)
            listOfFirst.append(fromNode[i]['first_name'])
            listOfLast.append(fromNode[i]['last_name'])
        return self.extract_names_logic(extract_text, listOfFirst, listOfFirst)

    def extract_names_logic(self, input_text, first=list, last=list):
        appendList = []
        lastName = []
        firstName = " "
        preProc = PreProcessing()
        tk = WhitespaceTokenizer()
        for txt in input_text:
            test = tk.tokenize(txt)
            for str in test:
                str = re.sub('[^A-Za-z0-9]+', '', str)
                appendList.append(str.lower())
            removeStopWords = preProc.removeStopWords(appendList)
        for token in removeStopWords:
            if (self.binarySearchAlgo(last, token)):
                lastName.append(token)

        return lastName

        # for i in range(len(fromNode)):
        #     print(fromNode[i]['school_id'])

    def extract_title(self, input_text):
        title = ''
        if len(input_text) > 0:
            # Extract the first item as the title
            title = input_text[0].strip()
        return title

    def extract_department(self, input_text):
        departments = [
            'School of Law', 'School of Business and Management',
            'School of Computer Studies', 'Senior High School',
            'School of Arts and Sciences', 'RITTC',
            'School of Allied Medical Sciences',
            'School of Engineering', 'School of Education'
        ]
        extracted_department = ''
        for text in input_text:
            for department in departments:
                if department in text:
                    extracted_department = department
                    break

            if extracted_department:
                break

        return extracted_department

    def binarySearchAlgo(self, listFromAlgo, search):
        listFromAlgo.sort()
        lower = [x.lower() for x in listFromAlgo]
        print(listFromAlgo)
        start = 0
        end = len(lower) - 1
        middle = (int)(end / 2)
        Found = False
        while (start <= end):
            middle = (int)(start + (end - start) / 2)
            if (lower[middle] == search):
                Found = True
                break
            elif (lower[middle] > search):
                end = middle - 1
            else:
                start = middle + 1
        return Found

    def extract_person(self, text_list):
        names = []
        name_formats = [
            "First Name Last Name",
            "Last Name, First Name",
            "First Name Middle Name Last Name",
            "Last Name, First Name Middle Name",
            "Last Name, First Name Middle Initial",
            "Title First Name Last Name",
            "First Name Middle Initial Last Name",
            "Last Name, First Initial Middle Initial"
        ]

        for text in text_list:
            # Extract names using NLTK
            nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
            for nltk_result in nltk_results:
                if type(nltk_result) == nltk.tree.Tree:
                    name = ''
                    for nltk_result_leaf in nltk_result.leaves():
                        name += nltk_result_leaf[0] + ' '
                    names.append(name.strip())

            # Extract names using SpaCy
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and ent.text not in names:
                    if len(ent.text.split()) > 1:
                        names.append(ent.text)

            # Extract names using regex based on name formats
            for name_format in name_formats:
                pattern = re.sub(r'[,\s]', r'\\s*', name_format)
                pattern = re.sub(r'First Initial', r'[A-Z]\.', pattern)
                pattern = re.sub(r'Middle Initial', r'[A-Z]', pattern)
                pattern = re.sub(r'Title', r'.+', pattern)
                matches = re.findall(pattern, text)
                names.extend(matches)

        return names

    def extract_published_date(self, input_text):
        extracted_date = None
        current_date = datetime.datetime.now()  # Get current date and time
        for text in input_text:
            # Extract date from text using regular expressions or other methods
            # Example: Extracts MM/YYYY format
            extracted_date_str = re.findall(r'\b\d{1,2}/\d{4}\b', text)
            if extracted_date_str:
                extracted_date = date_parser.parse(
                    extracted_date_str[0], fuzzy=True)

            # If extracted date is not found, try parsing the lines using date_formats
            if not extracted_date:
                date_formats = [
                    # Month day, Year (e.g. March 20, 2020)
                    '%B %d, %Y',
                    '%B %Y',             # Month Year (e.g. March 2020)
                    '%m/%Y',             # Month/Year (e.g. 03/2020)
                    # Abbreviated Month Year (e.g. Mar 2020)
                    '%b %Y',
                    # Month Year without slash (e.g. 03 2020)
                    '%m %Y',
                    'date %Y',           # Custom date format (e.g. date 2023)
                    # Custom date format (e.g. June 20, 2022)
                    '%B %d, %Y',
                    # Custom date format with time (e.g.June 20, 2022 12:34)
                    '%B %d, %Y %H:%M',
                ]
                for line in text.split('\n'):
                    try:
                        # Attempt to parse date from line using date_formats
                        for date_format in date_formats:
                            parsed_date = date_parser.parse(line, fuzzy=True, yearfirst=True,
                                                            default=current_date)
                            if parsed_date:
                                extracted_date = parsed_date
                                break
                        if extracted_date:
                            break
                    except ValueError:
                        pass

            if extracted_date:
                break  # If date is extracted from any text, break the loop

        if extracted_date:
            # Check if extracted month and year match the current date
            if extracted_date.month == current_date.month and extracted_date.year == current_date.year:
                # If yes, format as Month day, Year (e.g. March 9, 2023)
                if extracted_date.day > 9:
                    formatted_date = extracted_date.strftime(
                        "%B %d, %Y").replace(" 0", " ")
                else:
                    formatted_date = extracted_date.strftime("%B %d, %Y")
            else:
                # Otherwise, format as Month Year (e.g. March 2020)
                formatted_date = extracted_date.strftime("%B %d %Y")
            return formatted_date
        else:
            return None

    def main_DuplicateChecker(self):
        extractedText = " "
        finalText = " "
        count = 0
        extractPDF = Processing("")
        booleanValues = []
        isDuplicate = False
        txt_fromUpload = extractPDF.getFromPDF(self.document_path)
        preProssedFromUpload = self.preProcessing(txt_fromUpload)
        directory = (glob.glob("assets/upload/" + "/*.pdf"))
        for file in directory:
            print(file)
            with pdfplumber.open('rb', file) as pdf:
                for page in pdf.pages:  # just one page though, loop for future proofing
                    extractedText = page.extract_text()
                    finalText = finalText + extractedText
                    preProcessedFromLocal = self.preProcessing(finalText)
                    isDuplicate = self.duplicate_logic(
                        preProssedFromUpload, preProcessedFromLocal)
                    booleanValues.append(isDuplicate)
                    count += 1
                    break
            finalText = " "
        if (True in booleanValues):
            isDuplicate = True
        print(isDuplicate)
        return isDuplicate
        # self.duplicate_logic(uploaded_Text)

    def duplicate_logic(self, preProcessedFromUpload, preProcessedFromLocal):
        isDuplicate = False
        fromUpload = set(preProcessedFromUpload)
        fromLocal = set(preProcessedFromLocal)
        if (fromLocal == fromUpload):
            isDuplicate = True
        return isDuplicate

    def preProcessing(self, raw_Text):
        preProc = PreProcessing()
        raw_Text = preProc.removeSpecialCharacters(raw_Text)
        raw_Text = preProc.manual_tokenization(raw_Text)
        raw_Text = preProc.removeStopWords(raw_Text)
        raw_Text = preProc.toLowerCase(raw_Text)
        return raw_Text

    def calcualateRAKE(self, raw_text):
        # text = "Excel is an amazing tool for data science. But a lot of data science professionals do not like Excel."
        text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating set of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."
        raw_text = re.sub(r'[^a-zA-Z0-9\s]+', '', raw_text)
        tk = WhitespaceTokenizer()
        stop_words = set(stopwords.words('english'))
        test = tk.tokenize(raw_text)
        filtered_sentence = [w for w in test if not w.lower() in stop_words]

        unique = [x.lower() for x in filtered_sentence]
        phrases = self.getKeyPhrases(test, stop_words)
        word_frequency = self.getWordFrequency(unique)

        # features = list(word_frequency.keys())
        degree_of_word = self.getDegreeofWord(word_frequency, phrases)
        degree_score = self.getDegreeScore(
            word_frequency, degree_of_word, phrases)
        keyPhrases = self.extractKeyPhrases(phrases, degree_score)
        return keyPhrases

    def extractKeyPhrases(self, phrases, degree_score):
        totalDict = {}
        total = 0
        tk = WhitespaceTokenizer()
        for str in phrases:
            token = tk.tokenize(str)
            for x in degree_score:
                if x in token:
                    total = total + degree_score[x]
            totalDict[str] = total
            total = 0

        totalDictSorted = sorted(totalDict, key=totalDict.get, reverse=True)
        return totalDictSorted

    def getKeyPhrases(self, rawText, stopWords=set):
        count = 0
        phrases = []
        potential_phrase = ""
        for str in rawText:
            if str.lower() not in stopWords:
                potential_phrase = potential_phrase + " " + str.lower()
            else:
                phrases.append(potential_phrase)
                potential_phrase = ""
        new_phrases = [x for x in phrases if x != '']
        return new_phrases

    def getWordFrequency(self, uniqueWords):
        count = 0
        term_frequency = {}
        for unique in uniqueWords:
            if not unique in term_frequency:
                term_frequency[unique] = uniqueWords.count(unique)
        return term_frequency

    def getDegreeofWord(self, term_frequency, phrases):
        degreeOfWord = {}
        for t in term_frequency:
            degreeOfWord[t] = 0
        length = len(term_frequency)
        rows = length
        cols = length
        matrix = []
        # for i in range(rows):
        #     row = []
        #     for j in range(cols):
        #         row.append(0)
        #     matrix.append(row)
        # matrix = numpy.zeros((len(term_frequency)-1, len(term_frequency)-1))
        countRows = 0
        countCols = 0
        for i in term_frequency:
            for j in term_frequency:
                if (i == j):
                    matrix.append(self.getSimilar(i, term_frequency))
                else:
                    matrix.append(self.getNextToken(phrases, i, j))
                countCols += 1
            countRows += 1

        test = self.nest_list(matrix, length, length)
        i = 0
        listofDegree = []
        for x in test:
            listofDegree.append(sum(test[i]))
            i += 1
        j = 0
        for t in degreeOfWord:
            degreeOfWord[t] = listofDegree[j]
            j += 1
        return degreeOfWord
        # self.getNextToken(phrases)

    def getSimilar(self, text, term_frequency):
        if (text in term_frequency):
            return term_frequency[text]

    def getNextToken(self, keyPhrases, i, j):
        count = 0
        for phrases in keyPhrases:
            if i in phrases and j in phrases:
                count += 1
        return count

    def nest_list(self, list1, rows, columns):
        result = []
        start = 0
        end = columns
        for i in range(rows):
            result.append(list1[start:end])
            start += columns
            end += columns
        return result

    def getDegreeScore(self, term_frequency, degreeOfWord, phrases):
        degree_of_scores = {}

        for term in term_frequency:
            for degree in degreeOfWord:
                if (term == degree):
                    degree_of_scores[term] = degreeOfWord[degree] / \
                        term_frequency[term]
        return degree_of_scores
    # def main_checkDuplicate(self, extracted_txt):
    #     isDuplicate = False
    #     raw_Text = self.extract_forDuplicate()
    #     preProcessed_Text = self.preProcessing(raw_Text)
    #     isDuplicate = self.duplicate_checker(preProcessed_Text)
    #     return isDuplicate
        # with open("assets/upload/" + self.document_path, 'rb') as pdf_file:
        #     pdf_reader = pdfplumber.open(pdf_file)
        #     page = pdf_reader.pages[0]
        #     pdf_text = page.extract_text()
        #     print(pdf_text)
