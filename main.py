import os
import json
from flask import Flask, request
import pickle
import pandas as pd
import shap

# Créer une instance de l'application Flask
app = Flask(__name__)

# Lire les données CSV, charger le modèle et le seuil optimal
data_test = pd.read_csv("./data/application_test.csv", dtype={'SK_ID_CURR': str})
data_test_ohe = pd.read_csv("./data/application_test_ohe.csv", dtype={'SK_ID_CURR': str})
model_path = "./data/selected_model.pickle"
lgbm = pickle.load(open(model_path, 'rb'))
threshold_opt = 0.65
explainer = shap.Explainer(lgbm)
#explainer = pickle.load(open('./data/selected_model_explainer.pickle', 'rb'))


# Fonction réponse à la requête acceuil
@app.route("/")
def welkome():
    return "flask api running"

# Fonction réponse à la requête customer_data
@app.route('/customer_data/', methods=['GET'])
def customer_data():
    customer_id = request.args.get("customer_id")
    customer_row = data_test[data_test['SK_ID_CURR'] == str(customer_id)]
    response = {'customer_data': customer_row.to_json()}
    return json.dumps(response)

# Fonction réponse à la requête predict
@app.route('/predict/', methods=['GET'])
def predict():
    customer_id = request.args.get("customer_id")
    customer_row = data_test[data_test['SK_ID_CURR'] == str(customer_id)]
    if not customer_row.empty:
        customer_row_ohe = data_test_ohe.iloc[customer_row.index].drop(columns=['SK_ID_CURR'], axis=1)
        predictions = lgbm.predict_proba(customer_row_ohe)
        probability_negative_class = predictions[:, 1]
        if threshold_opt < probability_negative_class:
            classe = "refuse"
        else:
            classe = "accepte"
        response = {'negative_predict': probability_negative_class.tolist(),
                    'classe': classe}
        return json.dumps(response)

@app.route('/explain/', methods=['GET'])
def explain():
    customer_id = request.args.get("customer_id")
    customer_row_ohe = data_test_ohe[data_test['SK_ID_CURR'] == str(customer_id)].drop(columns=['SK_ID_CURR'], axis=1)
    customer_index = customer_row_ohe.index
    if not customer_row_ohe.empty:
        #customer_row_ohe = customers_data.iloc[customer_row.index].drop(columns=['SK_ID_CURR'], axis=1)
        #df_customer_row_ohe = pd.DataFrame(customer_row_ohe)#.transpose()
        #df_customer_row_ohe = df_customer_row_ohe.astype(float)
        shap_values = explainer.shap_values(customer_row_ohe)
        response = {'features_name': customer_row_ohe.columns.tolist(), 'shap_values': shap_values.tolist()}
        return json.dumps(response)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))