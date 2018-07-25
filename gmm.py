# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:56:43 2018

@author: mohit
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from get_embedding import sent_embedding
from sklearn.mixture import GMM

df = pd.read_csv('train.csv')
# Pre-Processing : remove punctations.
# digits lower case in sent_embediing.
# stop words left. use nltk for that
# decode left
df['data']= df['data'].apply(lambda x: x.translate(None, string.punctuation))
df['data']= df['data'].apply(lambda x: x.translate(None, string.digits))
# df['data']= df['data'].apply(lambda x: x.decode('utf-8'))
sentences = df['data'].tolist()
y = df['labels'].tolist()


X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)

gmm = GMM(n_components=2).fit(X_train)
labels = gmm.predict(X_test)
print(labels)
print('\n')
#probs = gmm.predict_proba(X)
#print(probs[:5].round(3))

probs = gmm.predict(X_test)
print(probs)

#with open(key+'.pickle', 'wb') as fo:
 #   pickle.dump(pl,fo)
