from information_extraction.document_extractor import DocumentExtractor
import re
import datetime
from dateutil import parser as date_parser
import random
from nameparser import HumanName
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
import spacy

# nltk.download('maxent_ne_chunker')
# nltk.download('words')

class InformationExtraction:
    def __init__(self, document_path):
        self.document_path = document_path
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_information(self):
        print(self.document_path)
        extractor = DocumentExtractor(self.document_path)
        extracted_text = extractor.extract_text_from_document()
        if extracted_text is not None:
            information = self.process_extracted_text(extracted_text)
            return information
        else:
            print('Invalid file format. Please upload a PDF file.')

    def process_extracted_text(self, input_text):
        information = {}

        information['title'] = self.extract_title(input_text)

        # Extract Department
        information['department'] = self.extract_department(input_text)
        
        information['author'] = self.extract_person(input_text)
        
        # information['adviser'] = self.extract_adviser(input_text)
        
        # Extract published date
        information['published_date'] = self.extract_published_date(input_text)

        # Return extracted information
        return information

    def extract_title(self, input_text):
        title = ''
        if len(input_text) > 0:
            title = input_text[0].strip()  # Extract the first item as the title
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
            extracted_date_str = re.findall(r'\b\d{1,2}/\d{4}\b', text)  # Example: Extracts MM/YYYY format
            if extracted_date_str:
                extracted_date = date_parser.parse(extracted_date_str[0], fuzzy=True)

            # If extracted date is not found, try parsing the lines using date_formats
            if not extracted_date:
                date_formats = [
                    '%B %d, %Y',         # Month day, Year (e.g. March 20, 2020)
                    '%B %Y',             # Month Year (e.g. March 2020)
                    '%m/%Y',             # Month/Year (e.g. 03/2020)
                    '%b %Y',             # Abbreviated Month Year (e.g. Mar 2020)
                    '%m %Y',             # Month Year without slash (e.g. 03 2020)
                    'date %Y',           # Custom date format (e.g. date 2023)
                    '%B %d, %Y',         # Custom date format (e.g. June 20, 2022)
                    '%B %d, %Y %H:%M',   # Custom date format with time (e.g.June 20, 2022 12:34)
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
                    formatted_date = extracted_date.strftime("%B %d, %Y").replace(" 0", " ")
                else:
                    formatted_date = extracted_date.strftime("%B %d, %Y")
            else:
                # Otherwise, format as Month Year (e.g. March 2020)
                formatted_date = extracted_date.strftime("%B %d %Y")
            return formatted_date
        else:
            return None