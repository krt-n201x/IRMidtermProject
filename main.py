import json
import string

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_and_clean(filepath):
    # https://pythonexamples.org/python-csv-to-json/
    # https://reactgo.com/python-remove-first-last-character-string/
    # https://milovantomasevic.com/blog/stackoverflow/2021-04-21-convert-csv-to-json-file-in-python/
    file = pd.read_csv(filepath)
    file = pd.DataFrame(file)
    df_new = file[file['Idiom'] == 'ENGLISH']
    df_new = df_new.drop_duplicates(subset=["SLink"])
    for i, row in df_new.iterrows():
        df_new.at[i, 'ALink'] = df_new.at[i, 'ALink'].lstrip('/').rstrip('/')
    return df_new
    # print("input word to do TF, TF-IDF, BM25")
    # inputword = input()
    # queryBM25(df_new,inputword)
    # tf(df_new, inputword)
    # tfidf(df_new, inputword)
    # searchArtis(df_new)
    # searchLyric(df_new)
    # df_new.to_json('data/lyrics.json', orient='records', indent=4)

def tf(df_new, inputword):
    # TF-IDF rank score
    print("TF is running...")
    # implement n-gram
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_new['Lyric'])
    # print n-gram
    # print(vectorizer.get_feature_names_out())
    print(X.shape)
    query = inputword
    query_vec = vectorizer.transform([query])
    results = cosine_similarity(X, query_vec).reshape((-1,))
    rank = 0
    tfJson = []


    for i in results.argsort()[-10:][::-1]:
        rank = rank + 1
        arthisname = df_new.iloc[i, 0]
        songname = df_new.iloc[i, 1]
        score = results[i]
        fulllyric = df_new.iloc[i, 3]
        # 5 first and last
        lyrics = df_new.iloc[i, 3]
        lyrics = lyrics.split()
        lyricsfirst = []
        lyricslast = []
        for j in range(5):
            # print(lyrics[j])
            lyricsfirst.append(lyrics[j])
        lastcount = 5
        for k in range(5):
            # print(lyrics[len(lyrics)-lastcount])
            lyricslast.append(lyrics[len(lyrics)-lastcount])
            lastcount = lastcount -1
        lastlyrics = ' '.join(lyricsfirst) + ' ... ' + ' '.join(lyricslast)

        tfJson.append([rank, arthisname, songname,fulllyric, lastlyrics, score])
        # print(lyrics)
        # print("Rank ", rank, " Score: ", results[i])
        # print(" Artis name: ", df_new.iloc[i, 0], " Song name: ", df_new.iloc[i, 1])

    tfJsondf = pd.DataFrame(tfJson, columns=["rank","artist","song","fulllyric","lyric","score"])
    tfJsondf.to_json('data/tfscore.json', orient='records', indent=4)


    print(tfJsondf.to_markdown(tablefmt="grid"))

    ####################################################################################
    ####################################################################################

class BM25():
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, ngram_range=(1,2))
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)


        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]


        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

def queryBM25(df_new,inputword) :
    print("BM25 is running...")
    data = df_new['Lyric']
    bm25lyrics = inputword
    bm25 = BM25()
    bm25.fit(data)
    result = bm25.transform(bm25lyrics, data)
    rank = 0
    bm25Json = []

    for i, index in enumerate(result.argsort()[-10:][::-1]):
        rank = rank + 1
        arthisname = df_new.iloc[index, 0]
        songname = df_new.iloc[index, 1]
        score = result[index]
        fulllyric = df_new.iloc[index, 3]
        # 5 first and last
        lyrics = df_new.iloc[index, 3]
        lyrics = lyrics.split()
        lyricsfirst = []
        lyricslast = []
        for j in range(5):
            # print(lyrics[j])
            lyricsfirst.append(lyrics[j])
        lastcount = 5
        for k in range(5):
            # print(lyrics[len(lyrics)-lastcount])
            lyricslast.append(lyrics[len(lyrics) - lastcount])
            lastcount = lastcount - 1
        lastlyrics = ' '.join(lyricsfirst) + ' ... ' + ' '.join(lyricslast)

        bm25Json.append([rank, arthisname, songname,fulllyric, lastlyrics, score])

    bm25Jsondf = pd.DataFrame(bm25Json, columns=["rank", "artist", "song","fulllyric", "lyric", "score"])
    bm25Jsondf.to_json('data/bm25score.json', orient='records', indent=4)

    print(bm25Jsondf.to_markdown(tablefmt="grid"))

def tfidf(df_new, inputword):
    # TF-IDF rank score
    print("TF-IDF is running...")
    # implement n-gram
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df_new['Lyric'])
    # print n-gram
    # print(vectorizer.get_feature_names_out())
    print(X.shape)
    query = inputword
    query_vec = vectorizer.transform([query])
    results = cosine_similarity(X, query_vec).reshape((-1,))
    rank = 0
    tfidfJson = []

    ngrams1Json = []
    ngrams2Json = []
    ngrams3Json = []
    for i in results.argsort()[-10:][::-1]:
        rank = rank + 1
        arthisname = df_new.iloc[i, 0]
        songname = df_new.iloc[i, 1]
        score = results[i]
        fulllyric = df_new.iloc[i, 3]
        lyrics = df_new.iloc[i, 3]

        # generate n-gram from top 10 lyrics

        lyrics1ngrams = ngrams(lyrics, 1)
        ngrams1Json.append([rank,songname,lyrics1ngrams])

        lyrics2ngrams = ngrams(lyrics, 2)
        ngrams2Json.append([rank, songname, lyrics2ngrams])

        lyrics3ngrams = ngrams(lyrics, 3)
        ngrams3Json.append([rank, songname, lyrics3ngrams])

        ########################################
        # 5 first and last
        lyrics = lyrics.split()
        lyricsfirst = []
        lyricslast = []
        for j in range(5):
            # print(lyrics[j])
            lyricsfirst.append(lyrics[j])
        lastcount = 5
        for k in range(5):
            # print(lyrics[len(lyrics)-lastcount])
            lyricslast.append(lyrics[len(lyrics)-lastcount])
            lastcount = lastcount -1
        lastlyrics = ' '.join(lyricsfirst) + ' ... ' + ' '.join(lyricslast)

        tfidfJson.append([rank, arthisname, songname,fulllyric, lastlyrics, score])
        # print(lyrics)
        # print("Rank ", rank, " Score: ", results[i])
        # print(" Artis name: ", df_new.iloc[i, 0], " Song name: ", df_new.iloc[i, 1])

    tfidfJsondf = pd.DataFrame(tfidfJson, columns=["rank","artist","song","fulllyric","lyric","score"])
    tfidfJsondf.to_json('data/tfidfscore.json', orient='records', indent=4)

    ngrams1Jsondf = pd.DataFrame(ngrams1Json, columns=["rank","song","ngram"])
    ngrams1Jsondf.to_json('data/ngram1.json', orient='records', indent=4)
    ngrams2Jsondf = pd.DataFrame(ngrams2Json, columns=["rank", "song", "ngram"])
    ngrams2Jsondf.to_json('data/ngram2.json', orient='records', indent=4)
    ngrams3Jsondf = pd.DataFrame(ngrams3Json, columns=["rank", "song", "ngram"])
    ngrams3Jsondf.to_json('data/ngram3.json', orient='records', indent=4)

    print(tfidfJsondf.to_markdown(tablefmt="grid"))

    ####################################################################################
    ####################################################################################


def ngrams(text, n):
    words = text.split()
    output = []
    for i in range(len(words) - n + 1):
        output.append(words[i:i + n])
    return output

def searchArtis(df_new, inputword):
    # search match the exact artist name
    print("Search All Song of the artis is running...")
    dataAname = df_new
    for i, row in dataAname.iterrows():
        dataAname.at[i, 'ALink'] = dataAname.at[i, 'ALink'].lower()
        dataAname.at[i, 'ALink'] = dataAname.at[i, 'ALink'].translate(
            str.maketrans('', '', string.punctuation + u'\xa0'))
        dataAname.at[i, 'ALink'] = dataAname.at[i, 'ALink'].translate(
            str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))

    print("In put Artis name: ")
    artisname = inputword
    cleaninput = artisname
    cleaninput = cleaninput.lower()
    cleaninput = cleaninput.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
    cleaninput = cleaninput.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    print("your clean input is: " + cleaninput)
    print(":: List of song name ::")
    song = []
    for i, row in dataAname.iterrows():
        if dataAname.at[i, 'ALink'] == cleaninput:
            song.append([dataAname.at[i, 'SName']])
            # print(dataAname.at[i, 'SName'])
    songdf = pd.DataFrame(song, columns=["SongName"])
    songdf = songdf.sort_values("SongName")
    nosong = 0
    tempJson = []
    for i, row in songdf.iterrows():
        nosong = nosong + 1
        tempJson.append([nosong,songdf.at[i, 'SongName']])
        print("No.", nosong, " : ", songdf.at[i, 'SongName'])

    df = pd.DataFrame(tempJson, columns=["no", "songname"])
    df.to_json('data/getartistsong.json', orient='records', indent=4)

def searchSongname(df_new,inputword):
    # search match the exact song name
    print("Search song is running...")
    dataSname = df_new
    for i, row in dataSname.iterrows():
        dataSname.at[i, 'SName'] = dataSname.at[i, 'SName'].lower()
        dataSname.at[i, 'SName'] = dataSname.at[i, 'SName'].translate(
            str.maketrans('', '', string.punctuation + u'\xa0'))
        dataSname.at[i, 'SName'] = dataSname.at[i, 'SName'].translate(
            str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))

    print("In put Song name: ")
    songname = inputword
    cleansonginput = songname
    cleansonginput = cleansonginput.lower()
    cleansonginput = cleansonginput.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
    cleansonginput = cleansonginput.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    print("your clean input is: " + cleansonginput)
    print(":: Lyrics ::")
    songLyrics = []
    for i, row in dataSname.iterrows():
        if dataSname.at[i, 'SName'] == cleansonginput:
            songLyrics.append([dataSname.at[i, 'Lyric'], dataSname.at[i, 'ALink'], dataSname.at[i, 'SName']])
    songLyricsdf = pd.DataFrame(songLyrics, columns=["Lyrics", "Artis","Song Name"])
    songLyricsdf = songLyricsdf.sort_values("Artis")
    nosong = 0
    tempJson = []
    for i, row in songLyricsdf.iterrows():
        nosong = nosong + 1
        tempJson.append([nosong,songLyricsdf.at[i, 'Artis'],songLyricsdf.at[i, 'Song Name'], songLyricsdf.at[i, 'Lyrics']])
        print("No.", nosong, " : ", " By Artis Name : ", songLyricsdf.at[i, 'Artis'], " Song name : ", songLyricsdf.at[i, 'Song Name'])
        print(" ======== ")
        print(songLyricsdf.at[i, 'Lyrics'])
        print(" ======== ")

    df = pd.DataFrame(tempJson, columns=["no","artist","songname","lyric"])
    df.to_json('data/getsong.json', orient='records', indent=4)

if __name__ == '__main__':
    get_and_clean('lyrics-data.csv')
