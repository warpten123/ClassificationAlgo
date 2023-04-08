import docx
import pdfplumber
import textract
import re
import PyPDF2

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

    def extract_paragraphs_from_text(self,input_text):
        # Split input text into paragraphs based on newline characters
        paragraphs = input_text.split('\n')

        # Initialize list to store processed paragraphs
        processed_paragraphs = []

        # Initialize variable to store current paragraph
        current_paragraph = ""

        # Loop through paragraphs and process them
        for paragraph in paragraphs:
            # If paragraph is not empty or whitespace-only
            if paragraph.strip():
                # If current paragraph is not empty, add space before next paragraph
                if current_paragraph:
                    current_paragraph += ' '

                # Append current paragraph with current line
                current_paragraph += paragraph.strip()
            else:
                # If current paragraph is not empty, add it to processed_paragraphs
                if current_paragraph:
                    processed_paragraphs.append(current_paragraph)
                    current_paragraph = ""

        # If current paragraph is not empty after loop, add it to processed_paragraphs
        if current_paragraph:
            processed_paragraphs.append(current_paragraph)

        # Return extracted paragraphs
        return processed_paragraphs

    def extract_text_from_document(self):
        if self.document_path.endswith('.pdf'):
            return self.extract_text_from_pdf()
        else:
            return None
