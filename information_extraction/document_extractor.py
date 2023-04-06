import docx
import pdfplumber
import textract
import re

class DocumentExtractor:
    def __init__(self, document_path):
        self.document_path = document_path

    def extract_text_from_pdf(self):
        with pdfplumber.open(self.document_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()
            if text:
                sentences = re.split(r'(?<=[!?])\s+|\n', text)
                sentences = [s for s in sentences if s]  # remove empty sentences
                return sentences
            else:
                return None

    def extract_text_from_document(self):
        if self.document_path.endswith('.pdf'):
            return self.extract_text_from_pdf()
        else:
            return None
