from flask import Flask, request, jsonify
from flask_cors import CORS
# import k_nearest_neighbor as knn
# import information_extraction as inform
# import TFIDF_FINAL as tfidf
import os
import sys

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from information_extraction.main import InformationExtraction
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={
            r"/returnAscii": {"origins": "*"}})


@app.route('/api', methods=['GET'])
def returnAscii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer
    # request.headers.add('Access-Control-Allow-Origin', '*')
    return d

@app.route('/upload-file', methods=['POST'])
def upload_file():
    # Extract file and research ID from request data
    file = request.files['file']
    research_id = request.form['research_id']
    print(file)
    print(research_id)

    # Generate a unique filename
    filename = f"{file.filename}"
    print(filename)

    # Save the uploaded file to /assets/upload directory
    file.save(os.path.join('assets', 'upload', filename))

    # Return success response
    return {'status': 'success', 'message': 'File uploaded successfully.'}

# Route for information extraction
@app.route('/python/information_extraction/<filename>', methods=['GET'])
def information_extraction_route(filename):
    # Instantiate InformationExtraction class
    info_extractor = InformationExtraction(filename)
    output = info_extractor.extract_information()
    
    if output is not None:
        print('Extracted Information:')
        print(output)

    # Return extracted information as JSON
    return jsonify(output)

# @app.route('/python/knn', methods=['GET'])
# def callKNN():
#     classification = knn.getResponse()
#     classification = str(request.args['algo'])
#     return classification





# @app.route('/python/addPDF', methods=['POST'])
# def addPDFToFlask():
#     # receive pdf from frontend
#     return ''


# def initDataSet():
#     init = tfidf.Processing()
#     init.createTFIDF()


if __name__ == "__main__":
    # initDataSet()
    app.run(debug=True)
