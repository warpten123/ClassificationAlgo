import docx
import pdfplumber
import textract
import re
import PyPDF2
import pytesseract
import os
from urllib.parse import quote, unquote
class DocumentExtractor:
    def __init__(self, document_path):
        self.document_path = document_path

    def extract_text_from_pdf(self):
        # Open PDF file
        with pdfplumber.open(self.document_path) as pdf:
            # Extract text from first page
            page = pdf.pages[0]
            extracted_text = page.extract_text()

        # Extract paragraphs from extracted text
        paragraphs = self.extract_paragraphs_from_text(extracted_text)

        # Print extracted paragraphs
        for i, paragraph in enumerate(paragraphs, 1):
            print(f'Paragraph {i}: {paragraph}')
        return paragraphs
    
    def extract_paragraphs_from_pdf(self):
        print(self.document_path)
        # Check if the file exists
        if os.path.exists(self.document_path):
            # Check if the file is readable
            if os.access(self.document_path, os.R_OK):
                # Open PDF file in binary mode
                with open(self.document_path, 'rb') as file:
                    # Create a PDF reader object
                    pdf_reader = PyPDF2.PdfReader(file)
                    # Extract text from all pages in the PDF
                    pdf_text = ''
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()

                    # Perform OCR using pytesseract
                    ocr_text = pytesseract.image_to_string(pdf_text)

                    # Extract paragraphs from OCR text
                    paragraphs = self.extract_paragraphs_from_text(ocr_text)

                return paragraphs
            else:
                print("File exists but is not readable.")
                return None
        else:
            print("File does not exist.")
            return None
    
    def extract_text_from_document(self):
        if self.document_path.endswith('.pdf'):
            return self.extract_paragraphs_from_pdf()
        else:
            return None
