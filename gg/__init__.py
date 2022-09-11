import logging
import sys,os
from pathlib import Path

# As PosixPath
sys.path.append(Path(__file__).parent)
print(sys.path)
import azure.functions as func

# My lib
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
import json
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


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')

    if user_id:
        user_id=int(user_id)
        results, recommended_cats = predict_best_category_for_user(user_id, model, articles_df)
        return func.HttpResponse(json.dumps({
                "articles": results,
                "categories": recommended_cats
            }))
    else:
        return func.HttpResponse(
             f"Votre requête a été lancée avec succès. Veuillez entrer votre identifiant utilisateur afin d'obtenir vos recommandations.",
             status_code=200
        )

