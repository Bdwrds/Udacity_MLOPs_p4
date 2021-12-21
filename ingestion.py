import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
ingest_record_file = config['ingest_record_file']
csv_output = config['csv_output']

fp_cwd = os.getcwd()
# create directory if doesnt exist
if os.path.isdir(output_folder_path) is not True:
    os.makedirs(os.path.join(fp_cwd, output_folder_path))

ingest_record = open(os.path.join(fp_cwd, output_folder_path,ingest_record_file), 'w')

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    ls_files = os.listdir(input_folder_path)
    df_comb = pd.DataFrame()
    for file in ls_files:
        if file[-3:] == 'csv':
            fp_df = os.path.join(fp_cwd, input_folder_path, file)
            df_sub = pd.read_csv(fp_df)
            df_comb.append(df_sub)
            ingest_record.write(fp_df + "\n")

    # drop duplicates
    df_comb = df_comb.drop_duplicates()

    # write csv to file
    fp_out = os.path.join(fp_cwd, output_folder_path, csv_output)
    df_comb.to_csv(fp_out)

    # save record of files ingested here
    ingest_record.close()

if __name__ == '__main__':
    merge_multiple_dataframe()
