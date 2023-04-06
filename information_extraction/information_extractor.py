from document_extractor import DocumentExtractor

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

    def match_keywords(self, lines, keywords):
        """
        Helper function to match keywords in a list of lines from extracted text.

        Args:
            lines (list): List of lines from extracted text.
            keywords (list): List of keywords to match.

        Returns:
            str: Extracted value for the matched keyword.
        """
        i = 0
        value = ""
        while i < len(lines):
            if any(keyword in lines[i].lower() for keyword in keywords):
                value += " " + lines[i].strip()
                i += 1
                while i < len(lines) and not any(keyword in lines[i].lower() for keyword in keywords):
                    value += " " + lines[i].strip()
                    i += 1
                break
            i += 1
        return value.strip()

    def process_extracted_text(self, extracted_text):
        """
        Process the extracted text to identify and store information.

        Args:
            extracted_text (list): List of lines containing the extracted text.

        Returns:
            dict: Dictionary containing extracted information.
        """
        # Split the extracted text into lines
        lines = extracted_text
        print(lines)
        # Process the extracted text to identify categories
        title_keywords = ['thesis', 'dissertation', 'research paper', 'project report']
        college_department_keywords = ['computer science', 'engineering', 'information technology']
        authors_keywords = ['by', 'authored by', 'written by']
        adviser_keywords = ['advised by', 'supervised by', 'Thesis Adviser']
        published_date_keywords = ['submitted on', 'date', 'published on']

        # Initialize variables to store extracted information
        title = ""
        college_department = ""
        authors = ""
        adviser = ""
        published_date = ""

        # Match keywords and store extracted values
        title = self.match_keywords(lines, title_keywords)
        college_department = self.match_keywords(lines, college_department_keywords)
        authors = self.match_keywords(lines, authors_keywords)
        adviser = self.match_keywords(lines, adviser_keywords)
        print(adviser)
        published_date = self.match_keywords(lines, published_date_keywords)

        # Return the extracted information as a dictionary
        return {
            'title': title,
            'college_department': college_department,
            'authors': authors,
            'adviser': adviser,
            'published_date': published_date
        }


document_path = 'EUL_ A Digital Research Repository System.pdf' 
ie = InformationExtraction(document_path)
information = ie.extract_information()
if information is not None:
    print('Extracted Information:')
    print(information)
