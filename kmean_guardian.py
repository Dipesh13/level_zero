import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=400, centers=4,
#                        cluster_std=0.60, random_state=0)
# X = X[:, ::-1] # flip axes for better plotting

import re
import pandas as pd
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn import preprocessing


df = pd.read_csv("train.csv")
# Pre-Processing : remove punctations and digits , stop words and lower case to be done in Tfidf
# df['data']= df['data'].apply(lambda x: x.translate(None, string.punctuation))
# df['data']= df['data'].apply(lambda x: x.translate(None, string.digits))
# df['data'] = df['data'].str.lower()
X= df['data']
y = df['labels']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)

pipe = Pipeline([
    ('vectorizer',TfidfVectorizer()),
    ('clf',KMeans(3, random_state=0))
])

pipe.fit(X)
p = pipe.predict(X_test)
print(" train accuracy: ", accuracy_score(y_test, p))

# plt.scatter(X, y, c=p, s=40, cmap='viridis');
# plt.scatter(X[:, 0], X[:, 1], c=p_t, s=40, cmap='viridis');
# plt.show()