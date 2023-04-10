from flask import Flask, request, jsonify
from flask_cors import CORS
# import k_nearest_neighbor as knn
# import information_extraction as inform
# import TFIDF_FINAL as tfidf
import os
import uuid

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

# @app.route('/python/knn', methods=['GET'])
# def callKNN():
#     classification = knn.getResponse()
#     classification = str(request.args['algo'])
#     return classification


# @app.route('/python/information_extraction', methods=['GET'])
# def callInformation():
#     extract = inform.Extract()
#     extract.autoNER()  # add pdf here


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
