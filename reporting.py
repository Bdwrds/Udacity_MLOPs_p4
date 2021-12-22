import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = config['test_data_path']
test_data_csv = config['test_data_csv']
output_model_path = config['output_model_path']

f_model = config['model_output']
fp_prod_path = config['prod_deployment_path']
fp_cwd = os.getcwd()
f_model = os.path.join(fp_cwd, fp_prod_path, f_model)
f_csv = os.path.join(fp_cwd, test_data_path, test_data_csv)

##############Function for reporting
def score_model(model_path, fp_csv, output_path):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df_data = pd.read_csv(fp_csv)
    X_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target = 'exited'
    X = df_data.loc[:, X_cols]
    y = df_data.loc[:, target]

    mdl = pickle.load(open(model_path, 'rb'))
    preds = model_predictions(mdl, fp_csv)
    cm = metrics.confusion_matrix(y, preds)
    print(cm)

    # if latest version of sklearn
    #cm_plot = metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds)

    fig = plt.figure()
    cm_plot = metrics.plot_confusion_matrix(mdl, X, y)
    plt.title('Confusion Matrix on Test Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.path.join(output_model_path, 'confusionMatrix.png'))

if __name__ == '__main__':
    score_model(f_model, f_csv, output_model_path)
