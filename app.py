from flask import Flask, jsonify, request
import pickle
from diagnostics import model_predictions, dataframe_summary, dataframe_missing, execution_time, outdated_packages_list
from scoring import score_model
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])

fp_cwd = os.getcwd()
f_model = config['model_output']
fp_prod = config['prod_deployment_path']
f_model_prod = os.path.join(fp_cwd, fp_prod, f_model)

prediction_model = pickle.load(open(f_model_prod, 'rb'))

@app.route("/", methods=['GET', 'OPTIONS'])
def index():
    response = request.args.get("check_working")
    return str(response)

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    csv_path = request.args.get('csv_path')
    preds = model_predictions(prediction_model, csv_path)
    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats1():
    csv_path = request.args.get('csv_path')
    f1_score = score_model(f_model_prod, csv_path, fp_scores=None)
    return str(round(f1_score, 4))

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats2():
    csv_path = request.args.get('csv_path')
    ls_summary = dataframe_summary(csv_path)
    ls_json = [str(summary) for summary in ls_summary]
    return jsonify(ls_json)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def stats3():
    csv_path = request.args.get('csv_path')
    ls_na = dataframe_missing(csv_path)
    ls_time = execution_time()
    ls_output = outdated_packages_list()
    return jsonify([ls_na, ls_time, ls_output.to_json()])

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
