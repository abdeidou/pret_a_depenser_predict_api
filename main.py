import os
import io
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import shap
import matplotlib


from flask import Flask, request, send_file
from flask_caching import Cache
from flask import jsonify
matplotlib.use('Agg')

# Lire les données CSV, charger le modèle et le seuil optimal
data_test = pd.read_csv("./data/application_test.csv",
                        dtype={'SK_ID_CURR': str})
data_test_ohe = pd.read_csv("./data/application_test_ohe.csv",
                            dtype={'SK_ID_CURR': str})
model_path = "./data/selected_model.pickle"
lgbm = pickle.load(open(model_path, 'rb'))
threshold_opt = 0.65
X = data_test_ohe.drop(columns=['SK_ID_CURR'], axis=1)
features = X.columns.tolist()
explainer = shap.Explainer(lgbm)
shap_values = explainer.shap_values(X)
explainer_tree = shap.TreeExplainer(lgbm)
shap_values_tree = explainer_tree.shap_values(X)
# Créer un objet Explanation
explanation = shap.Explanation(values=shap_values_tree,
                               base_values=explainer_tree.expected_value,
                               data=X)


# Créer une instance de l'application Flask
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


# Fonction réponse à la requête acceuil
@app.route("/")
def welkome():
    return "flask api running"

# Fonction réponse à la requête customer_data


@app.route('/customer_data/', methods=['GET'])
def customer_data():
    customer_id = request.args.get("customer_id")
    cache_key = f"customer_data_{customer_id}"
    customer_row = data_test[data_test['SK_ID_CURR'] == str(customer_id)]
    response = {'customer_data': customer_row.to_json()}
    return jsonify(response)


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
        return jsonify(response)

@app.route('/explain_local/', methods=['GET'])
def explain_local():
    customer_id = request.args.get("customer_id")
    # Mise en cache
    cache_key = f"explain_local_{customer_id}"
    # Vérifier si la réponse est mise en cache
    cached_response = cache.get(cache_key)
    if cached_response is not None:
        return cached_response
    # Générer le graphique SHAP pour le client spécifié
    customer_row_ohe = data_test_ohe[data_test['SK_ID_CURR'] == str(customer_id)]
    customer_index = customer_row_ohe.index
    shap.waterfall_plot(explanation[int(customer_index.values[0])], show=False)
    # Save plot to BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=250, format="png", bbox_inches='tight')
    plt.close()
    # Rewind BytesIO
    buffer.seek(0)
    response = send_file(buffer, mimetype='image/png')

    # Mettre la réponse en cache
    cache.set(cache_key, response, timeout=300)
    return response


@app.route('/explain_global/')
def explain_global():
    cached_response = cache.get('explain_global')
    if cached_response is not None:
        return cached_response
    # Créer le graphique SHAP beeswarm
    #shap.plots.beeswarm(shap_values)
    # Créer le graphique SHAP
    shap.summary_plot(shap_values, X)
    # Save plot to BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=250, format="png")
    plt.close()
    # Rewind BytesIO
    buffer.seek(0)
    response = send_file(buffer, mimetype='image/png')
    # Mettre la réponse en cache
    cache.set('explain_global', response, timeout=300)
    return response


@app.route('/position/', methods=['GET'])
def position():
    customer_id = request.args.get("customer_id")
    feature = request.args.get("variable")
    customer_value = data_test_ohe.loc[data_test_ohe['SK_ID_CURR'] == str(customer_id), feature].values[0]
    # Calculer le maximum et le minimum des autres clients
    customers_max_value = data_test_ohe.loc[data_test_ohe['SK_ID_CURR'] != str(customer_id), feature].max()
    customers_min_value = data_test_ohe.loc[data_test_ohe['SK_ID_CURR'] != str(customer_id), feature].min()
    response = {'customer_value': customer_value,
                'customers_min_value': customers_min_value,
                'customers_max_value': customers_max_value}
    return jsonify(response)

@app.route('/feature_names')
def feature_names():
    response = {'feature_names': features}
    return jsonify(response)

@app.route('/threshold')
def threshold():
    response = {'threshold': threshold_opt}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    #gunicorn -w 4 -b 0.0.0.0:8080 main:app

