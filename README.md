# pret_a_depenser_predict_api

Une API de “scoring crédit” pour prédir d'accorder ou réfuser un crédit à un client en utilisant un classifieur LigntGBM optimisé 
et s'appuyant sur des sources de données test variées (données comportementales, données provenant d'autres institutions financières, etc.)

Réalisée en Flask et se basant sur ce repo à l'adresse:

https://credit-predict-2olkar52da-ew.a.run.app

Le service est deployé sur Google cloud.

Exemple de test:

https://credit-predict-2olkar52da-ew.a.run.app/customer_data/?customer_id=100028

https://credit-predict-2olkar52da-ew.a.run.app/predict/?customer_id=100028


## Les dossiers

data:

-> application_test.csv: les données clients test

-> application_test_ohe.csv: les données clients test traitées par One_Hot_Encoding (disponibilité des données pour le modèle)
  
-> best_model.pickle: le modèle optimisé LightGBM

-> IDOUMOHMED_ABDELAAZIZ_2_notebook_modélisation_012024.ipynb: notebook de modélisation 
  
sources:

-> main.py: le code source du service flask
  
tests:

-> test_api: script test de l'api en pytest, intégration continue, GitHub Action
  
.idea: dossier de configuration PyCharm

.github/workflows:

-> tests.yml: services d'intégration continue lance le script pytest à chaque commit

-> deploy.yml: services de deploiement continue lance le dockerfile à chaque commit en ajoutant dans le commit 'to deploy'
  
requirements.txt: liste des packages requis pour le projet

.dockerignore, .gcloudignore, .igitignore: les composants à exclure

## Utilisation

Ce projet a été créé dans le cadre de formation profesionnelle et ouvert à des fins d'évaluation.
