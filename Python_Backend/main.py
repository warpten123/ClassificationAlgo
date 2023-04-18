


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import requests
import glob
from collections import OrderedDict
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.insert(0, 'cosine-similarity\cosine-similarity.py')
from tfidf.extraction_helper import Helper
from information_extraction.main import InformationExtraction
from knn.k_nearest_neighbor import KNN
from tfidf.TFIDF_FINAL import Processing
uri = 'http://127.0.0.1:3000'
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
    result = {}
    filename = f"{file.filename}"
    path_directory = "assets/upload"
    # CAN ADD CHECKER HERE IN UPLOAD WHAT'S THE BETTER WAY? LAHI OR I SAGOL NALANG ARI? FOR NOW LAHI LANG NGA ENDPOINT
    file.save(os.path.join('assets', 'upload', filename))
    # helper = Helper()
    # go = helper.acceptanceChecker(filename)
    # if (go != True):
    #     result = {'status': 'failed',
    #               'message': 'Chosen FIle Does Not Meet the Standard Requirement'}
    # else:
    #     file.save(os.path.join('assets', 'upload', filename))
    #     result = {'status': 'success',
    #               'message': 'File Uploaded Successfullu!'}

    # file.save(os.path.join('assets', 'upload', filename))
    # if not any(os.scandir(path_directory)):
    #     file.save(os.path.join('assets', 'upload', filename))
    #     result = {'status': 'success',
    #               'message': 'File uploaded successfully.'}
    # else:
    #     file.save(os.path.join('assets', 'temp', filename))
    #     info_extractor = InformationExtraction(filename)
    #     output = info_extractor.extract_information()
    #     if output is not None:
    #         go = info_extractor.main_DuplicateChecker()
    #         if (go == False):
    #             result = {'status': 'success',
    #                       'message': 'File uploaded successfully.'}
    #             file.save(os.path.join('assets', 'upload', filename))
    #         else:
    #             result = {'status': 'failed', 'message': 'Duplicate File'}

    # file.save(os.path.join('assets', 'temp', filename))
    # # Save the uploaded file to /assets/upload directory

    # info_extractor = InformationExtraction(filename)
    # output = info_extractor.extract_information()
    # if output is not None:
    #     go = info_extractor.main_DuplicateChecker()
    #     if(go == False):
    #         result = {'status': 'success', 'message': 'File uploaded successfully.'}
    #         file.save(os.path.join('assets', 'upload', filename))
    #     else:
    #         result = {'status': 'failed', 'message': 'Duplicate File'}

    return result

# Route for information extraction


@app.route('/python/information_extraction/<filename>', methods=['GET'])
def information_extraction_route(filename):
    # Instantiate InformationExtraction class
    info_extractor = InformationExtraction(filename)
    fromNode = getDataFromNode()
    # for i in range(len(fromNode)):
    #     print(fromNode[i]['school_id'])

    output = info_extractor.extract_information(fromNode)
    return jsonify(output)


# @app.route('/python/information_extraction/<filename>', methods=['GET'])
# def information_extraction_route(filename):
#     # Instantiate InformationExtraction class
#     info_extractor = InformationExtraction(filename)
#     output = info_extractor.extract_information()

#     if output is not None:
#         go = info_extractor.main_DuplicateChecker()

#     # Return extracted information as JSON
#     return jsonify(go)


@app.route('/python/knn', methods=['GET'])
def callKNN():
    classification = KNN()
    accuracy = classification.main()

    return jsonify(accuracy)


@app.route('/python/tfidf/<filename>', methods=['GET'])
def getTFIDF(filename):
    tfidf = Processing(" ")
    rawtext = tfidf.insertNewData(filename)
    return {'RESULT': rawtext}


@app.route('/python/node', methods=['GET'])
def getDataFromNode():
    response = requests.get(uri + '/api/account/getNames/listNames')
    response_json = response.json()
    return response_json[0]


@app.route('/python/knn/extract_forDataSet/<filename>', methods=['GET'])
def extractForDataSet(filename):
    helper = Helper()
    tfidf = Processing(" ")
    result = helper.main_logic(filename)
    # information = InformationExtraction(filename)
    # keyPhrases = information.calcualateRAKE(result['appendedData'])
    # tfidf.insertNewData(result)
    return jsonify(result)


@app.route('/python/knn/check_acceptance/<filename>', methods=['GET'])
def checkAcceptance(filename):
    helper = Helper()
    result = helper.acceptanceChecker(filename)
    # tfidf.insertNewData(result)
    return {'result': result}


@app.route('/python/knn/test_cosine', methods=['GET'])
def test_cosine():
    helper = Helper()
    result = helper.acceptanceChecker()
    # tfidf.insertNewData(result)
    return {'result': result}


@app.route('/python/information_extractor/keyphrases/<filename>', methods=['GET'])
def key_phrases(filename):
    information = InformationExtraction(filename)
    result = information.calcualateRAKE()
    # tfidf.insertNewData(result)
    return {'result': result}


def checkDataSet():
    cont = False
    csv = "TFIDF.csv"
    directory = (glob.glob("tfidf/Results/" + "/*.csv"))
    for file in directory:
        if (file == "tfidf/Results\TFIDF.csv"):
            cont = True
    return cont


def initializeDataSet():
    tfidf = Processing(" ")
    tfidf.createTFIDF(" ")


@app.before_request
def before_first_request_func():
    if (checkDataSet() != True):
        initializeDataSet()


if __name__ == "__main__":

    # check if data set has been cooked or not
    # if data set has been cooked, ignoreq
    # if not create TFIDF.
    # check if file TFIDF.csv exists in folder Results
    before_first_request_func()
    app.run(debug=True)
