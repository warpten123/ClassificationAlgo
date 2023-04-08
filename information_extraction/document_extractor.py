import docx
import pdfplumber
import textract
import re
import PyPDF2
import pytesseract
import os
from urllib.parse import quote, unquote
from PIL import Image
import io
class DocumentExtractor:
    def __init__(self, document_path):
        self.document_path = document_path

    def extract_text_from_pdf(self):
        # Open PDF file in binary mode
        with open(self.document_path, 'rb') as pdf_file:
            # Create a PDFPlumber object
            pdf_reader = pdfplumber.open(pdf_file)
            # Extract text from the first page of the PDF
            page = pdf_reader.pages[0]
            pdf_text = page.extract_text()

            # Convert PDF page to image
            x0, y0, x1, y1 = page.cropbox or (0, 0, page.width, page.height)
            image = page.to_image(resolution=300)
            image_file = 'temp_image.png'
            image.save(image_file, format='png')
            
            # Specify the path to Tesseract executable
            pytesseract.pytesseract.tesseract_cmd = r"E:\\Programs\\tes\\tesseract.exe"

            # Perform OCR using pytesseract
            ocr_text = pytesseract.image_to_string(image_file)

            # Extract paragraphs from extracted text
            paragraphs = self.extract_paragraphs_from_text(ocr_text)

            # Print extracted paragraphs
            for i, paragraph in enumerate(paragraphs, 1):
                print(f'Paragraph {i}: {paragraph}')

            # Delete temporary image file
            os.remove(image_file)

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
        
    def extract_paragraphs_from_text(self, input_text):
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
