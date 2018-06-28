import pickle

with open('oneclass_tfidf_.pickle', 'rb') as fi:
    model = pickle.load(fi)

def prediction(list_pred):
    for row in list_pred:
        return (model.predict([row])[0])
