# encoding: utf-8
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath,get_tmpfile

def sent_embedding(sentence):
    # print(sentence)
    sent_emb = np.mean([get_embeddings(sent) for sent in sentence],axis=0)
    return sent_emb

def get_embeddings(word):
    glove_file = datapath('/home/dipesh/Downloads/glove.6B.50d.txt')
    tmp_file = get_tmpfile("glove_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    return model.wv[word]

# here text = row . iterating over all the rows of text column one by one
def preprocess(text):
    """
    to do 1) if no words found return unk token in order to get embeddings of unk
    :param text:
    :return:
    """
    text = text.lower()
    doc = word_tokenize(text)
    # doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    embedding = sent_embedding(doc)
    return embedding


df = pd.read_csv('train.csv',encoding="utf-8")
dataframe = df[df['labels'] == 'football']

dataframe['features'] = dataframe['data'].apply(preprocess)
dataframe.to_csv('temp.csv')

X = dataframe['features']
y = dataframe['labels']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)

pl = Pipeline([
    ('clf',svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))
])

pl.fit(X_train,y_train)

preds = pl.predict(X_train)
print(" train accuracy: ", accuracy_score(y_train, preds))
preds_test = pl.predict(X_test)
print(" test accuracy: ", accuracy_score(y_test, preds_test))

with open('oneclass_football_emb.pickle', 'wb') as fo:
    pickle.dump(pl,fo)