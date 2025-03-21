import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

    # shampoo_df = data.iloc[:35501]
    # conditioner_df = data.iloc[35501:67892]
    # oil_df = data.iloc[67892:]

cond_min_index = 35501
oil_min_index = 67892

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file) #list of dictionaries
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])

#each datapoint has "product_name", "star", "content", and "profile" (might be blank)
data = pd.DataFrame(data) #pandas DataFrame
data['content'] = data['content'].map(lambda x: x.strip())
data['modcontent'] = data['profile'] + ' ' + data['content'] #PREpend profile info (even if blank)
# data['content'] = data['content'].map(lambda x: x.strip()) #could strip again but IDC
# data = data.drop('profile', axis=1) #no longer needed in the demo

vec = TfidfVectorizer(max_df=0.8, min_df=10)
tdidf_matrix = vec.fit_transform(data['modcontent']).toarray()
data = data.drop('modcontent', axis=1) #no longer needed in the demo


    # shampoo_df = data.iloc[:35501]
    # conditioner_df = data.iloc[35501:67892]
    # oil_df = data.iloc[67892:]

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
# def json_search(query):
#     matches = []
#     merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
#     matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
#     matches_filtered = matches[['title', 'descr', 'imdb_rating']]
#     matches_filtered_json = matches_filtered.to_json(orient='records')
#     return matches_filtered_json

def reviews_json_search(wantPoo, wantCond, wantOil, query):
    #Thanks Bisola!
    query_vec = vec.transform([query]).toarray().T
    poo_matches = '[]'
    cond_matches = '[]'
    oil_matches = '[]' #if actual arrays, JS gets mad when parsing
    if wantPoo:
        poo_docs = tdidf_matrix[:cond_min_index]
        poo_docs = np.dot(poo_docs, query_vec).flatten() #shoud have len(poo_docs)
        # poo_docs = np.argmax(poo_docs) #now just a single index
        poo_docs = np.argsort(poo_docs)[:-6:-1] #top 5 in descending order: -1 to -5 inclusive
        #TODO AFTER DEMO: change this to top 5 or top 10 reviews per product, not just 1
        # also means change HTML to expect LIST instead of single
        poo_matches = data.iloc[poo_docs].to_json(orient='records')
        # print(type(poo_matches))
        # poo_matches.append(data.iloc[poo_docs].to_json(orient='records'))
        print(poo_matches)
    #TODO: conditioner and oil
    

    #return should be dict or list of dicts;
    #JS will interpret at actual JSON
    return {'shampoos': poo_matches, 'conditioners': cond_matches, 'oils': oil_matches}


@app.route("/")
def home():
    # return render_template('base.html',title="sample html")
    return render_template('demo.html',title="demo html")

@app.route("/reviews")
def reviews_search():
    query = request.args.get("query")
    wantPoo = request.args.get("wantPoo")
    wantCond = request.args.get("wantCond")
    wantOil = request.args.get("wantOil")
    return reviews_json_search(wantPoo, wantCond, wantOil, query)
    # return reviews_json_search(text)

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)