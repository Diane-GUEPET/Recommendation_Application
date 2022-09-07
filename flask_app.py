# Importation des librairies
from email.mime import application
import os
import json

import random

import pandas as pd
import numpy as np
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import missingno
import requests
from flask import Flask, jsonify, request, render_template
import sklearn
import sklearn.model_selection

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise import KNNWithMeans, SVD
from surprise import accuracy

import pickle

#Settings
pd.set_option('display.max_columns', None)
sns.set(color_codes=True)

#To be avoided in your notebook.
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)

# Chargement du fichier d'articles
def load_files():
    articles_df = pd.read_csv("articles_metadata.csv")
    return articles_df

articles_df = load_files()

# Chargement du modèle
def load_models():
    pkl_filename = "./models/pickle_surprise_model_KNNWithMeans.pkl"
    with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
    return model

model = load_models()

# Fonction de prédiction
def predict_best_category_for_user(user_id, model, articles_df):
    predictions = {}
    
    #Category 1 to 460
    for i in range(1, 460):
        _, cat_id, _, est, err = model.predict(user_id, i)
        
        #Keep prediction only if we could keep it.
        if (err != True):
            predictions[cat_id] = est
    
    best_cats_to_recommend = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
    
    recommended_articles = []
    for key, _ in best_cats_to_recommend.items():
        recommended_articles.append(int(articles_df[articles_df['category_id'] == key]['article_id'].sample(1).values))
    
    #return random_articles_for_best_cat, best_cat_to_recommend
    return recommended_articles, best_cats_to_recommend


# default access page

@app.route("/")
def main():
    return render_template('index.html')

# upload selected user_id and forward to predictions

@app.route('/predict', methods=['POST'])
def predict():
    for x in request.form.values():
        user_id=int(x)
    # user_id=str(x)
    # user_id=int(user_id)
    # args=request.args
    # user_id=args.get('user_id')
    # user_id=int(user_id)
    #print(args)
    # # parse input features from request
    # request_json = request.get_json()
    # x = str(request_json['input'])
    # Résultats pour l'utilisateur 1    
    results, recommended_cats = predict_best_category_for_user(user_id, model, articles_df)
    #predict_cat= 'Catégories recommandées : {recommended_cats}'
    #predict_art = '5 articles tirés au hasard de la meilleure catégorie {results}'
    text1=recommended_cats
    text2=results
    return render_template("index.html"
    , pred_cat_text="Meilleure(s) catégorie(s) pour l'utilisateur {}: {}\n".format(user_id, text1)
    , pred_art_text= "\n Articles recommandés pour l'utilisateur {}: {}".format(user_id, text2))
    


# @app.route('/predict_art', methods=['POST'])
# def predicted():
#     args=request.args
#     user_id=args.get('user_id')
#     #print(args)
#     # # parse input features from request
#     # request_json = request.get_json()
#     # x = str(request_json['input'])
#     # Résultats pour l'utilisateur 1    
#     results, recommended_cats = predict_best_category_for_user(user_id, model, articles_df)
#     #predict_cat= 'Catégories recommandées : {recommended_cats}'
#     #predict_art = '5 articles tirés au hasard de la meilleure catégorie {results}'
#     text=results
#     return print_result(text)


# # retrieve file from 'static/images' directory
# @app.route('/results')
# def print_result(filename):
#     return print(filename)


if __name__=='__main__':
    app.run(port=8080, debug=True)