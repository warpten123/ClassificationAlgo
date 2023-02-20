from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api', methods=['GET'])
def returnAscii():
    d = {}
    inputchr = str(request.args['query'])
    answer = str(ord(inputchr))
    d['output'] = answer
    return d


if __name__ == "__main__":
    app.run(debug=True)
