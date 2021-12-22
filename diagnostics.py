import subprocess
import pandas as pd
import timeit
import os
import json
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])

fp_cwd = os.getcwd()

f_ingest_csv = config['ingest_record_file']
f_dataset_csv = os.path.join(fp_cwd, dataset_csv_path, f_ingest_csv)

fp_test_data = config['test_data_path']
f_test_data = config['test_data_csv']

f_model = config['model_output']
fp_prod = config['prod_deployment_path']

f_model_prod = os.path.join(fp_cwd, fp_prod, f_model)
f_data = os.path.join(fp_cwd, fp_test_data, f_test_data)

mdl = pickle.load(open(f_model_prod, 'rb'))

##################Function to get model predictions
def model_predictions(model, df_path):
    #read the deployed model and a test dataset, calculate predictions
    df = pd.read_csv(df_path)
    X_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = df.loc[:, X_cols]
    preds = model.predict(X)
    return preds
    #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(fp):
    df = pd.read_csv(fp)
    #calculate summary statistics here
    all_stats = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            all_stats.append(df[col].mean())
            all_stats.append(df[col].median())
            all_stats.append(df[col].min())
            all_stats.append(df[col].max())
            #col_stats = [col, df[col].mean(), df[col].median(), df[col].min(), df[col].max()]
            #all_stats.append(col_stats)
    return all_stats #return value should be a list containing all summary statistics

def dataframe_missing(fp):
    df = pd.read_csv(fp)
    na_prop = [df[col].isna().sum()/ len(df.index) for col in df.columns]
    return na_prop

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    ls_ingest_timing = []
    st = timeit.default_timer()
    subprocess.run(['python', 'ingestion.py'])
    ls_ingest_timing.append(timeit.default_timer()-st)

    ls_train_timing = []
    st = timeit.default_timer()
    subprocess.run(['python', 'training.py'])
    ls_train_timing.append(timeit.default_timer()-st)
    return [ls_ingest_timing,ls_train_timing] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    ls_install = subprocess.check_output(['pip', 'list', '--outdated'])
    with open('installed.txt', 'w') as fl:
        fl.write(ls_install.decode("utf-8"))
    df_pip = pd.read_csv('installed.txt', sep=r'\s+', skiprows=[1])
    fl.close()
    os.remove('installed.txt')

    ls_frozen = []
    with open('requirements.txt', 'r') as frozen:
        for line in frozen.readlines():
            ls_frozen.append(line.split("=="))
    df_frozen = pd.DataFrame(ls_frozen, columns=['Package', 'Frozen'])
    df_comb = pd.merge(df_pip, df_frozen, on='Package')
    return df_comb


if __name__ == '__main__':
    model_predictions(mdl, f_data)
    dataframe_summary(f_dataset_csv)
    dataframe_missing(f_dataset_csv)
    execution_time()
    outdated_packages_list()





    
