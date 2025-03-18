import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

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
    #each datapoint has "product_name", "star", "content", and "profile" (might be blank)
    shampoo_df = pd.DataFrame(data[:35501])
    conditioner_df = pd.DataFrame(data[35501:67892])
    oil_df = pd.DataFrame(data[67892:])

    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])

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

def reviews_json_search():
    #TODO: man
    return


@app.route("/")
def home():
    # return render_template('base.html',title="sample html")
    return render_template('demo.html',title="demo html")

@app.route("/reviews")
def reviews_search():
    text = request.args.get("query")
    return json_search(text)
    # return reviews_json_search(text)

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)