from document_extractor import DocumentExtractor
import spacy
from dateutil.parser import parse as date_parse
class InformationExtraction:
    def __init__(self, document_path):
        self.document_path = document_path
        
    def extract_information(self):
        extractor = DocumentExtractor(self.document_path)
        extracted_text = extractor.extract_text_from_document()
        if extracted_text is not None:
            information = self.process_extracted_text(extracted_text)
            return information
        else:
            print('Invalid file format. Please upload a PDF file.')

    def word_tokenize(text):
        words = []
        for sentence in text:
            words.extend(sentence.split())
        return words

    def pos_tag(tokens):
        tokens = text.split() # split text into words
        pos_tags = []
        for token in tokens:
            if token in ['A', 'An', 'The', 'Of', 'Thesis', 'Adviser', 'Date']:
                pos_tags.append((token, 'DT'))
            elif token.endswith(':'):
                pos_tags.append((token[:-1], 'NNP'))
                pos_tags.append((':', ':'))
            elif token in [',', '.']:
                pos_tags.append((token, 'PUNCT'))
            else:
                pos_tags.append((token, 'NN'))
        return pos_tags
    
    def process_extracted_text(self, input_text):
        print(input_text)
        for i, paragraph in enumerate(input_text, 1):
            print(f'Paragraph {i}: {paragraph}')
        extracted_info = {}

        # Extract title
        title = input_text[0]
        extracted_info['title'] = title if title else None

        # Extract college/department
        for line in input_text:
            if 'Faculty of' in line:
                extracted_info['college_department'] = line.split('Faculty of')[-1].strip()
                break

        # Combine input text into a single string
        input_text = ' '.join(input_text)

        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")

        # Process input text with spaCy
        doc = nlp(input_text)

        # Extract title
        extracted_info['title'] = doc.ents[0].text if doc.ents else None

        # Extract college/department
        for sent in doc.sents:
            if "Faculty of" in sent.text:
                extracted_info['college_department'] = sent.text.replace("Faculty of", "").strip()
                break

        # Extract authors
        extracted_info['authors'] = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        print(extracted_info['authors'][0])

        # Extract adviser
        extracted_info['adviser'] = None
        if extracted_info['authors']:
            adviser = extracted_info['authors'].pop()
            extracted_info['adviser'] = adviser

        # Extract published date
        extracted_info['published_date'] = self.extract_published_date(input_text)

        return extracted_info

    
    def extract_published_date(self, input_text):
        extracted_date = None

        # Extract published date using date parsing library
        for line in input_text.split('\n'):
            try:
                # Attempt to parse date from line
                parsed_date = date_parse(line, fuzzy=True)
                if parsed_date:
                    extracted_date = parsed_date.strftime("%B %Y")
                    break
            except ValueError:
                pass

        return extracted_date if extracted_date else None
    
document_path = 'Final_RECall_CS_Thesis_Paper.pdf' 
ie = InformationExtraction(document_path)
information = ie.extract_information()
if information is not None:
    print('Extracted Information:')
    print(information)
