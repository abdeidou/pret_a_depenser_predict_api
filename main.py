import os
from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello YouTube!! I hope you liked it"

@app.route('/double', methods=['GET'])
def multiple():
    number = request.args.get("number")
    result = number * 2
    return str(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# test github action