# -*- coding: utf-8 -*-
from flask import Flask, request
from flask_cors import CORS
import requests
import json
import pickle
from predict import prediction

app = Flask(__name__)
CORS(app)

model_file = './oneclass_tfidf_.pickle'
with open(model_file, 'rb') as f:
	model = pickle.load(f)

@app.route('/predict',methods=['POST'])
def predict_tfidf():
    """
    1) predict end point to accept label from user and load model based on the label , eg football
    2) train end point which accept columns
    3) upload end point for train.csv
    4) predict end point to accept files and folders.
    5) replicate this for other classifiers and models.
    :return:
    """
    data = json.loads(request.data.decode('utf8'))
    preds = []
    for val in data['articles']:
        preds.append(prediction(val))
    return json.dumps({'label' : preds})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug = True)