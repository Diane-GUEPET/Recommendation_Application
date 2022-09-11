# Importation des librairies

import requests
from flask import Flask, jsonify, request, render_template
import requests

import warnings
warnings.filterwarnings("ignore")
AZURE_FUNCTION_URL = "https://first-recomm-api.azurewebsites.net/api/gg"



app = Flask(__name__)

# default access page


@app.route("/")
def main():
    return render_template('index.html')

# upload selected user_id and forward to predictions


@app.route('/predict', methods=['POST'])
def predict():
    # print(request.form.values()[0])
    # user_id=int(request.form.values)
    for x in request.form.values():
        user_id = int(x)
        print(user_id)

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'user_id': user_id}

    # sending get request and saving the response as response object
    r = requests.get(url=AZURE_FUNCTION_URL, params=PARAMS)

    # extracting data in json format
    data = r.json()
    # print(data)
    return render_template("index.html", pred_cat_text="Meilleure(s) catégorie(s) pour l'utilisateur {}: {}\n".format(user_id, data["categories"]), pred_art_text="\n Articles recommandés pour l'utilisateur {}: {}".format(user_id, data["articles"]))


if __name__ == '__main__':
    app.run(port=8080, debug=True)
