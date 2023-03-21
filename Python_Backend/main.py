from flask import Flask, request, jsonify
import k_nearest_neighbor as knn
import information_extraction as inform
import TFIDF_FINAL as tfidf


app = Flask(__name__)


@app.route('/api', methods=['GET'])
def returnAscii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer
    return d


@app.route('/python/knn', methods=['GET'])
def callKNN():
    classification = knn.getResponse()
    classification = str(request.args['algo'])
    return classification


@app.route('/python/information_extraction', methods=['GET'])
def callInformation():
    extract = inform.Extract()
    extract.autoNER()  # add pdf here


@app.route('/python/addPDF', methods=['POST'])
def addPDFToFlask():
    # receive pdf from frontend
    return ''


def initDataSet():
    init = tfidf.Processing()
    init.createTFIDF()


if __name__ == "__main__":
    initDataSet()
    app.run(debug=True)
