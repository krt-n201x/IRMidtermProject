import json

def getsearchSongname():
    songresult = open('data/getsong.json', "r")
    jsonContent = songresult.read()
    data = json.loads(jsonContent)
    return data

def getsearchArtis():
    songresult = open('data/getartistsong.json', "r")
    jsonContent = songresult.read()
    data = json.loads(jsonContent)
    return data

def getBM25():
    bm25result = open('data/bm25score.json', "r")
    jsonContent = bm25result.read()
    data = json.loads(jsonContent)
    return data

def gettf():
    tfresult = open('data/tfscore.json', "r")
    jsonContent = tfresult.read()
    data = json.loads(jsonContent)
    return data

def gettfidf():
    tfidfresult = open('data/tfidfscore.json', "r")
    jsonContent = tfidfresult.read()
    data = json.loads(jsonContent)
    return data