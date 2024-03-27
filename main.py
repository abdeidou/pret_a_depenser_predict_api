import os
from flask import Flask, request

app = Flask(__name__)


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