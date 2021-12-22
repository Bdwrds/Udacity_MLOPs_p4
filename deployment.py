import os
import json
from shutil import copyfile

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

ls_ingest = config['ingest_record_file']
f_model = config['model_output']
f_score_file = config['score_file']
fp_model = config['output_model_path']

fp_cwd = os.getcwd()
fp_model_deploy_path = os.path.join(fp_cwd, prod_deployment_path)
fp_model_train = os.path.join(fp_cwd, fp_model, f_model)
fp_model_score = os.path.join(fp_cwd, fp_model, f_score_file)
fp_ingest_data = os.path.join(fp_cwd, dataset_csv_path, ls_ingest)

if os.path.isdir(fp_model_deploy_path) is not True:
    os.makedirs(fp_model_deploy_path)

####################function for deployment
def store_model_into_pickle(fp_model_deploy_path, fp_model_train, fp_model_score, fp_ingest_data):
    f_ingest_deploy = os.path.join(fp_model_deploy_path, os.path.basename(fp_ingest_data))
    f_model_deploy = os.path.join(fp_model_deploy_path, os.path.basename(fp_model_train))
    f_model_score = os.path.join(fp_model_deploy_path, os.path.basename(fp_model_score))
    copyfile(fp_ingest_data, f_ingest_deploy)
    print(f"Copied: {fp_ingest_data} to prod")
    copyfile(fp_model_train, f_model_deploy)
    print(f"Copied: {fp_model_train} to prod")
    copyfile(fp_model_score, f_model_score)
    print(f"Copied: {fp_model_score} to prod")

if __name__=="__main__":
    store_model_into_pickle(fp_model_deploy_path, fp_model_train, fp_model_score, fp_ingest_data)