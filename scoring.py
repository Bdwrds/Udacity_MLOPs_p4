from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
test_data_csv = config['test_data_csv']
fp_score_file = config['score_file']

model_path = os.path.join(config['output_model_path'])
model_output = config['model_output']
fp_cwd = os.getcwd()
fp_model_path = os.path.join(fp_cwd, model_path)
fp_model = os.path.join(fp_model_path, model_output)
fp_csv = os.path.join(fp_cwd, test_data_path, test_data_csv)
fp_scores = os.path.join(fp_cwd, model_path, fp_score_file)

#################Function for model scoring
def score_model(fp_model, fp_csv, fp_scores):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    X_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'
    df_test = pd.read_csv(fp_csv)
    X_test = df_test.loc[:, X_cols]
    y_test = df_test.loc[:, target]

    mdl = pickle.load(open(fp_model, 'rb'))

    y_pred = mdl.predict(X_test)
    f1_score = metrics.f1_score(y_pred, y_test)
    print(f"F1 Score: {f1_score}")

    with open(fp_scores, 'w+') as score_file:
        score_file.write(str(f1_score))

if __name__ == "__main__":
    score_model(fp_model, fp_csv, fp_scores)