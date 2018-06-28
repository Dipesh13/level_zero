import pandas as pd
import json
import requests
import argparse

parser = argparse.ArgumentParser(description='extract tables from excel files')
parser.add_argument('-filename', help='path to an excel file')
args = parser.parse_args()
filename = args.filename

def evaluate(filename):
    df = pd.read_csv(filename)
    labels = df.label.unique()

    for label in labels:
        list_data = []
        dict_data = {"articles":[]}
        test_df = df['data'][df['labels']==label]
        for row_index,row in test_df.iteritems():
            list_data.append(row)
            dict_data["articles"].append(row)
        # print(label)

        res = requests.post('http://0.0.0.0:5000/predict',data=json.dumps(dict_data))
        preds = res.json()['prediction']
        df_out = pd.DataFrame({'query':list_data,
                           'predictions':preds})
        df_out.to_csv(label+'.csv',index=False)

if __name__ == '__main__':
    evaluate(filename)