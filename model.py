import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score


def train_model(dataframe,label=''):
    if label :
        dataframe = df[df['labels'] == label]
        X = dataframe['data']
        y = dataframe['labels']
    else:
        X = dataframe['data']
        y = dataframe['labels']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)

    pl = Pipeline([
        ('vectorizer',TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,max_df=100,max_features=None)),
        ('clf',svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1))
    ])

    pl.fit(X_train,y_train)

    preds = pl.predict(X_train)
    print(" train accuracy: ", accuracy_score(y_train, preds))
    preds_test = pl.predict(X_test)
    print(" test accuracy: ", accuracy_score(y_test, preds_test))

    with open('oneclass_tfidf_'+label+'.pickle', 'wb') as fo:
        pickle.dump(pl,fo)


# colnames=['data','label']
df = pd.read_csv('train.csv')
# label = 'football'
# train_model(df,label)
train_model(df)