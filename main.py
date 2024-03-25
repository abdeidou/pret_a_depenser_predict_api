import json
from flask import Flask, request
import pandas as pd
import pickle
from waitress import serve

# Créer une instance de l'application Flask
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello"

# Lancer le processus flask
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
#test