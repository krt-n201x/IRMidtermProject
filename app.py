from flask import Flask, request, jsonify

import allget
import main

app = Flask(__name__)


@app.route('/getsong', methods=['GET'])
def getSongname():
    inputword = request.args.get('inputword')
    main.searchSongname(df_new, inputword)
    data = allget.getsearchSongname()
    if (data == []):
        return errormgs
    if (data != []):
        return jsonify(data)


@app.route('/getartist', methods=['GET'])
def getArtis():
    inputword = request.args.get('inputword')
    main.searchArtis(df_new, inputword)
    data = allget.getsearchArtis()
    if (data == []):
        return errormgs
    if (data != []):
        return jsonify(data)


@app.route('/getbm25', methods=['GET'])
def getbm25():
    inputword = request.args.get('inputword')
    main.queryBM25(df_new, inputword)
    data = allget.getBM25()
    if (data == []):
        return errormgs
    if (data != []):
        return jsonify(data)


@app.route('/gettf', methods=['GET'])
def gettf():
    inputword = request.args.get('inputword')
    main.tf(df_new, inputword)
    data = allget.gettf()
    if (data == []):
        return errormgs
    if (data != []):
        return jsonify(data)


@app.route('/gettfidf', methods=['GET'])
def gettfidf():
    inputword = request.args.get('inputword')
    main.tfidf(df_new, inputword)
    data = allget.gettfidf()
    if (data == []):
        return errormgs
    if (data != []):
        return jsonify(data)


@app.route('/search', methods=['POST', 'GET'])
def search():
    body = request.get_json()
    print(body['score'])
    print(body['query'])
    if body['score'] == 'tf':
        main.tf(df_new, body['query'])
        data = allget.gettf()
        return jsonify(data)
    if body['score'] == 'tf-idf':
        main.tfidf(df_new, body['query'])
        data = allget.gettf()
        return jsonify(data)
    if body['score'] == 'bm25':
        main.queryBM25(df_new, body['query'])
        data = allget.gettf()
        return jsonify(data)
    else:
        return errormgs


# python app.py
if __name__ == '__main__':
    df_new = main.get_and_clean('lyrics-data.csv')
    errormgs = {"errormgs": "do not have data"}
    app.run(debug=True)
