import os
import json
from flask import Flask, request, send_file
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
shap_values = explainer.shap_values(X)
explainer_tree = shap.TreeExplainer(lgbm)
shap_values_tree = explainer_tree.shap_values(X)
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


import matplotlib.pyplot as plt
import io
import base64
from flask import jsonify
@app.route('/explain_local/', methods=['GET'])
def explain_local():
    customer_id = request.args.get("customer_id")
    customer_row_ohe = data_test_ohe[data_test['SK_ID_CURR'] == str(customer_id)]
    customer_index = customer_row_ohe.index
    # Créer un objet Explanation
    explanation = shap.Explanation(values=shap_values_tree, base_values=explainer_tree.expected_value, data=X)
    # Obtenez l'index du client et créez le graphique SHAP
    shap.waterfall_plot(explanation)
    # Enregistrer le graphique dans un buffer mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Convertir le graphique en base64
    graph_data = base64.b64encode(buf.read()).decode('utf-8')
    # Créer la réponse JSON avec les données du graphique
    response = {'shap_plot': graph_data}
    return jsonify(response)

@app.route('/explain_global')
def explain_global():
    # Créer le graphique SHAP beeswarm
    #shap.plots.beeswarm(shap_values)
    # Créer le graphique SHAP
    shap.summary_plot(shap_values, X)

    # Enregistrer le graphique dans un buffer mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convertir le graphique en base64
    graph_data = base64.b64encode(buf.read()).decode('utf-8')

    # Créer la réponse JSON avec les données du graphique
    response = {'shap_plot': graph_data}

    return jsonify(response)


@app.route('/threshold')
def threshold():
    response = {'threshold': threshold_opt}
    return json.dumps(response)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))