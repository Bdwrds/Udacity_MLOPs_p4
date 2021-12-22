"""
The main section for updating and deploying new models
author: Ben E
date: 21/12/22
"""
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import subprocess
import json
import os
import sys
import pickle

# read the main config file
with open('config.json') as cf:
    config = json.load(cf)

fp_cwd = os.getcwd()

## read various parameters
fp_prod = config['prod_deployment_path']
f_ingested = config['ingest_record_file']
fp_new_data = config['input_folder_path']
csv_output = config['csv_output']
f_score_file = config['score_file']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']
fp_output_model_path = config['output_model_path']
f_model = config['model_output']

fp_model_deploy_path = os.path.join(fp_cwd, prod_deployment_path)
f_model_prod = os.path.join(fp_cwd, fp_prod, f_model)
fp_model_train = os.path.join(fp_cwd, fp_output_model_path, f_model)
fp_csv = os.path.join(fp_cwd, output_folder_path, csv_output)
fp_model_score = os.path.join(fp_cwd, fp_output_model_path, f_score_file)
fp_ingest_data = os.path.join(fp_cwd, output_folder_path, f_ingested)

# get a list of any new files
ls_data_files = os.listdir(fp_new_data)

# read the previous files
with open(os.path.join(fp_prod, f_ingested), 'r') as f:
    prev_files = f.read().splitlines()

# read the previous score for later comparison
with open(os.path.join(fp_prod, f_score_file), 'r') as f:
    prev_score = f.readlines()

# convert from list and str
prev_score = float(prev_score[0])

# are files new
new_files = [new_f for new_f in ls_data_files if new_f not in prev_files and new_f[-3:] == 'csv']


##################Deciding whether to proceed, part 1
print("CHECK NEW FILES")
if not new_files:
    print("No new files - exit script")
    sys.exit()
else:
    print("Found new files:", new_files)
    print("INGESTION STEP")
    ingestion.merge_multiple_dataframe(os.path.join(fp_cwd, fp_new_data))

##################Checking for model drift
print("LOAD MODEL")
mdl = pickle.load(open(f_model_prod, 'rb'))
preds = diagnostics.model_predictions(mdl, fp_csv)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
print("DRIFT SECTION")
f1_score = scoring.score_model(f_model_prod, fp_csv, fp_scores=None)

if f1_score >= prev_score:
    print(f"New score is higher: {f1_score} >= {prev_score}")
else:
    print(f"New score is lower: {f1_score} < {prev_score}")
    sys.exit()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
print("TRAIN MODEL")
training.train_model(fp_csv, fp_model_train)
_ = scoring.score_model(fp_model_train, fp_csv, fp_model_score)
print("DEPLOY MODEL")
deployment.store_model_into_pickle(fp_model_deploy_path, fp_model_train, fp_model_score, fp_ingest_data)


##################Diagnostics and reporting
print("REPORTING ON MODEL")
reporting.score_model(f_model_prod, fp_csv, fp_output_model_path)
print("API RUN & REPORTING")
test_response = 1
response = subprocess.run(["curl", "127.0.0.1:8000/?check_working="+str(test_response)], capture_output=True).stdout
if not response.decode('utf-8'):
    print("API not running currently..")
elif int(response.decode('utf-8')) == test_response:
    subprocess.run(['python', 'apicalls.py'])
else:
    print("API return incorrect results..")
print("END OF SCRIPT")


