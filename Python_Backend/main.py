


import json
import time
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
import os
import sys
import requests
import glob
import nltk
from collections import OrderedDict


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from knn.tfidf_only_confusion import ONLY
from knn.cosine import Cosine
from tfidf.extraction_helper import Helper
from information_extraction.main import InformationExtraction
from knn.k_nearest_neighbor import KNN
from tfidf.TFIDF_FINAL import Processing
from knn.testing import Testing
sys.path.insert(0, 'cosine-similarity/cosine-similarity.py')
sys.path.insert(0, 'knn/testing.py')
sys.path.insert(0, 'information_extraction/testing_extraction.py')
uri = 'http://127.0.0.1:3000'
# uri = 'http://192.168.143.57:3000'
app = Flask(__name__)
# run_with_ngrok(app)
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
    return result

# Route for information extraction


@app.route('/python/information_extraction/<filename>', methods=['GET'])
def information_extraction_route(filename):
    # Instantiate InformationExtraction class

    info_extractor = InformationExtraction(filename)
    # fromNode = getDataFromNode()
    # for i in range(len(fromNode)):extractAllPDF
    #     print(fromNode[i]['school_id'])

    output = info_extractor.extract_information()
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

@app.route('/python/knn/testing/', methods=['GET'])  # AUTOMATED TESTING SA KNN
def getKNN():
    classification = KNN()
    # helper = Helper()
    # data = helper.main_logic(filename)
    start_time = time.time()
    classification.automated_testing()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Final Execution Time (KNN k = 17):", execution_time, "seconds")
    return "test"


# LEGIT NGA KNN I SPECIFY ANG VALUE SA K, KANANG SECOND PARAMTER
@app.route('/python/knn/scatter_plot/<filename>', methods=['GET'])
def scatter_plot(filename):
    classification = KNN()
    helper = Helper()
    data = helper.main_logic(filename)
    result = classification.testing(data['appendedData'], 1)
    return result


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

    # str = ' '.join(keyPhrases)
    # tfidf.insertNewData(result)
    return result


@app.route('/python/knn/check_acceptance/<filename>', methods=['GET'])
def checkAcceptance(filename):
    helper = Helper()
    result = helper.acceptanceChecker(filename)
    print(result)
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
    newList = []
    helper = Helper()
    result = helper.main_logic(filename)
    information = InformationExtraction(filename)
    keyPhrases = information.calcualateRAKE(result['appendedData'])
    for i in range(5):
        newList.append(keyPhrases[i])
    # str = ','.join(newList)
    return newList


def checkDataSet():
    cont = False
    csv = "TFIDF.csv"
    directory = (glob.glob("tfidf/Results/" + "/*.csv"))
    for file in directory:
        if (file == "tfidf/Results/TFIDF.csv"):
            cont = True
    return cont


def initializeDataSet():
    tfidf = Processing(" ")
    tfidf.createTFIDF(" ")


# TFIDF + COSINE i nvm na ang knn calssifier, cosine gihapon na
@app.route('/python/classify/<filename>', methods=['GET'])
def classify(filename):
    helper = Helper()
    cosine = Cosine()
    knn = KNN()
    appendedData = helper.main_logic(filename)
   
    data = cosine.classifyResearch(appendedData['appendedData'], False)
    # predict = knn.knn_classifier(data, 1)
    print("FINALE BITCH",data)
    sorted_dict = dict(sorted(data.items(), key=lambda item: item[1]))

    # str = ','.join(newList)
    return sorted_dict


@app.route('/python/testing_matrix/<filename>', methods=['GET'])
def matrix(filename):
    cosine = Cosine()
    return cosine.get_cosine_matrix(filename)


@app.route('/python/testing_TFIDF/<filename>', methods=['GET'])  # TFIDF ONLY
def tfidf(filename):
    testing = ONLY()
    helper = Helper()
    cosine = Cosine()
    knn = KNN()
    appendedData = helper.main_logic(filename)
    result = testing.getTFIDF(appendedData['appendedData'])
    return result


# TESTING FOR TFIDF + COSINE
@app.route('/python/testing/cosine', methods=['GET'])
def accuracy():
    testing = Testing()
    return testing.extractAllPDF()


@app.before_request
def before_first_request_func():
    if (checkDataSet() != True):
        initializeDataSet()


if __name__ == "__main__":
    nltk.download('punkt_tab')
    # check if data set has been cooked or not
    # if data set has been cooked, ignoreq
    # if not create TFIDF.
    # check if file TFIDF.csv exists in folder Results
    before_first_request_func()
    app.run(debug=False)
