import os
import json
from flask import Flask, request
import pickle
import pandas as pd

# Créer une instance de l'application Flask
app = Flask(__name__)

# Fonction charger le modèle
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Lire les données CSV, charger le modèle et le seuil optimal
data_test = pd.read_csv("./data/application_test.csv", dtype={'SK_ID_CURR': str})
data_test_ohe = pd.read_csv("./data/application_test_ohe.csv", dtype={'SK_ID_CURR': str})
customers_data = data_test
customers_data_ohe = data_test_ohe
model_path = "./data/best_model.pickle"
lgbm = load_model(model_path)
threshold_opt = 0.3

# Fonction réponse à la requête customer_data
#@app.route('/customer_data', methods=['GET'])
@app.route('/customer_data/', methods=['GET'])
def customer_data():
    customer_id = request.args.get("customer_id")
    customer_row = customers_data[customers_data['SK_ID_CURR'] == str(customer_id)]
    response = {'customer_data': customer_row.to_json()}
    return json.dumps(response)

# Fonction réponse à la requête predict
@app.route('/predict/', methods=['GET'])
@app.route('/predict')
def predict():
    customer_id = request.args.get("customer_id")
    customer_row = customers_data[customers_data['SK_ID_CURR'] == str(customer_id)]
    if not customer_row.empty:
        customer_row_ohe = customers_data_ohe.iloc[customer_row.index].drop(columns=['SK_ID_CURR'], axis=1)
        predictions = lgbm.predict_proba(customer_row_ohe).tolist()
        response = {'customer_predict': predictions}
        return json.dumps(response)

# Fonction réponse à la requête threshold
@app.route('/threshold')
def threshold():
    return str(threshold_opt)

@app.route("/")
def hello():
    return "flask api"

@app.route('/test', methods=['GET'])
def test():
    return "/test"

@app.route('/double', methods=['GET'])
@app.route('/double/', methods=['GET'])
def double():
    params = request.args.get("params")
    try:
        number = int(params)
        result = number * 2
        return str(result)
    except ValueError:
        return "Invalid input: Please provide a valid number."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# test github action