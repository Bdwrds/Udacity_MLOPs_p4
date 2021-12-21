# from flask import Flask, session, jsonify, request
import pandas as pd
# import numpy as np
import pickle
import os
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
csv_output = os.path.join(config['csv_output'])
model_output = config['model_output']
fp_cwd = os.getcwd()
fp_csv = os.path.join(fp_cwd, dataset_csv_path, csv_output)
fp_model_path = os.path.join(fp_cwd, model_path)
fp_model = os.path.join(fp_model_path, model_output)

if os.path.isdir(fp_model_path) is not True:
    os.makedirs(fp_model_path)

#################Function for training the model
def train_model(fp_csv, fp_model):
    # load the data
    df_final = pd.read_csv(fp_csv)

    # refine the data
    X_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'
    X = df_final.loc[:, X_cols]
    y = df_final.loc[:, target]

    # split the test and train
    # X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    # use this logistic regression for training
    mdl = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    mdl.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(mdl, open(fp_model, 'wb'))

if __name__=="__main__":
    train_model(fp_csv, fp_model)