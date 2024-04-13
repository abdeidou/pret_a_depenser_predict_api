import os
import json
from flask import Flask, request, make_response
import pickle
import pandas as pd
import numpy as np
import shap
import zlib
# Créer une instance de l'application Flask
app = Flask(__name__)

# Lire les données CSV, charger le modèle et le seuil optimal
data_test = pd.read_csv("./data/application_test.csv", dtype={'SK_ID_CURR': str})
data_test_ohe = pd.read_csv("./data/application_test_ohe.csv", dtype={'SK_ID_CURR': str})
model_path = "./data/selected_model.pickle"
lgbm = pickle.load(open(model_path, 'rb'))
threshold_opt = 0.65
X = data_test_ohe.drop(columns=['SK_ID_CURR'], axis=1)
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


from functools import lru_cache
import jsonify
@app.route('/explain')
@lru_cache(maxsize=128)  # Taille maximale du cache
def explain_cached():
    # Votre code pour générer les SHAP values
    shap_values = explainer.shap_values(X)
    response = {'shap_values': shap_values.tolist()}
    return jsonify(response)

@app.route('/explain_local/', methods=['GET'])
def explain_local():
    customer_id = request.args.get("customer_id")
    customer_row_ohe = data_test_ohe[data_test['SK_ID_CURR'] == str(customer_id)].drop(columns=['SK_ID_CURR'], axis=1)
    customer_index = customer_row_ohe.index
    if not customer_row_ohe.empty:
        #customer_row_ohe = customers_data.iloc[customer_row.index].drop(columns=['SK_ID_CURR'], axis=1)
        #df_customer_row_ohe = pd.DataFrame(customer_row_ohe)#.transpose()
        #df_customer_row_ohe = df_customer_row_ohe.astype(float)

        #shap_values_local = explainer.shap_values(customer_row_ohe)
        #response = {'feature_names': customer_row_ohe.columns.tolist(), 'shap_values_local': shap_values_local.tolist()}
        #return json.dumps(response)
        shap_values_local = explainer.shap_values(customer_row_ohe)
        feature_names = customer_row_ohe.columns.tolist()
        response = {'shap_values_local': shap_values_local.tolist(), 'feature_names': feature_names}
        return json.dumps(response)

@app.route('/explain_global')
def explain_global():
    #X = data_test_ohe.drop(columns=['SK_ID_CURR'], axis=1)
    #shap_values_global = explainer.shap_values(X)
    #feature_importance = np.abs(shap_values_global).mean(axis=0)
    #sorted_inds = np.argsort(feature_importance)
    #top_inds = sorted_inds[-10:]
    #top_feature_names = X.columns[top_inds].tolist()
    #top_shap_values_global = shap_values_global[top_inds].tolist()
    #response = {'top_feature_names': top_feature_names, 'top_shap_values_global': top_shap_values_global}
    #return json.dumps(response)
    shap_values_global = explainer.shap_values(X)
    top_n = 10  # Change this to desired number of top SHAP values
    abs_shap_values = np.abs(shap_values_global)
    top_indices = np.argsort(abs_shap_values.flatten())[-top_n:]
    top_shap_values_global = shap_values_global[:, top_indices]
    response = {'top_shap_values_global': top_shap_values_global}
    return json.dumps(response)


@app.route('/threshold')
def threshold():
    response = {'threshold': threshold_opt}
    return json.dumps(response)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))