import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
# from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

    # shampoo_df = data.iloc[:35501]
    # conditioner_df = data.iloc[35501:67892]
    # oil_df = data.iloc[67892:]

# cond_min_index = 35501
# oil_min_index = 67892
cond_min_index = 298
oil_min_index = 540

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'init.json')

json_file_path = os.path.join(current_directory, 'dataset', 'ulta_reviews.json')
vocab_file_path = os.path.join(current_directory, 'dataset', 'ulta_vocabulary.json')
index_to_id_file_path = os.path.join(current_directory, 'dataset', 'ulta_product_ids.json')
products_file_path = os.path.join(current_directory, 'dataset', 'products_ulta.json')
allergies_file_path = os.path.join(current_directory, 'dataset', 'allergy_map.json')


# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file) #list of strings, all preprocessing done except embedding
    
    #list of dictionaries
#     # episodes_df = pd.DataFrame(data['episodes'])
#     # reviews_df = pd.DataFrame(data['reviews'])

# data = [review['profile'] + ' ' + review['content'] for review in data]
with open(vocab_file_path, 'r') as f:
    vocab = json.load(f) # ngram to index dictionary
# vec = TfidfVectorizer(max_df=0.8, min_df=10, ngram_range=(1,3))
vec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1,2))
# vocab = None
vec.fit(data)
# data = None
# with open(json_file_path, 'r') as file:
#     data = json.load(file) #list of dictionaries
# with open('svd.npy', 'rb') as f:
with open('svd_NEWDATA_U.npy', 'rb') as f:
    docs_compressed_normalized = np.load(f)
with open('svd_NEWDATA_V001.npy', 'rb') as f:
    words_compressed_normalized = np.load(f)
with open('svd_NEWDATA_V002.npy', 'rb') as f:
    words_compressed_normalized = np.vstack((words_compressed_normalized,np.load(f)))


#each datapoint has "product_name", "star", "content", and "profile" (might be blank)
# data = pd.DataFrame(data) #pandas DataFrame
# data['content'] = data['content'].map(lambda x: x.strip())
# data['modcontent'] = data['profile'] + ' ' + data['content'] #PREpend profile info (even if blank)
# data['content'] = data['profile'] + '$$' + data['content'] #PREpend profile info (even if blank)
# data['content'] = data['content'].map(lambda x: x.strip()) #could strip again but IDC
# data = data.drop('profile', axis=1) #no longer needed in the demo

# vec = TfidfVectorizer(max_df=0.8, min_df=10)
# # tdidf_matrix = vec.fit_transform(data['content']) 
# tdidf_matrix = vec.fit_transform(data['modcontent'])
# data = data.drop('modcontent', axis=1) #no longer needed in the demo


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

def reviews_json_search(wantPoo, wantCond, wantOil, query, bad_idx):
    #Thanks Bisola!
    # print("wantPoo: " + wantPoo)
    # print("wantCond: " + wantCond)
    # print("wantOil: " + wantOil)
    # print(type(wantPoo)) #THEY'RE STRINGS
    # print(type(wantCond))
    # print(type(wantOil))
    # query_vec = vec.transform([query]) #TOARRAY
    with open(index_to_id_file_path, 'r') as f:
        index_to_id = json.load(f)
    with open(products_file_path, 'r') as f:
        products_info = json.load(f)

    query_vec = vec.transform([query]).toarray() 
    print(query_vec)
    query_vec = normalize(np.dot(query_vec, words_compressed_normalized)).squeeze()
    poo_matches = '[]'
    cond_matches = '[]'
    oil_matches = '[]' #if actual arrays, JS gets mad when parsing
    if wantPoo:
        bad_idx_poo = [x for x in bad_idx if x<cond_min_index]
        poo_docs = docs_compressed_normalized[:cond_min_index]
        poo_docs = np.delete(poo_docs, bad_idx_poo, axis=0)

        poo_docs = np.dot(poo_docs, query_vec)
        print(poo_docs[0:5])
        poo_docs = np.argsort(-poo_docs)[:5] #sort descending, then get first 5
        print(poo_docs)
        # poo_matches = data.iloc[poo_docs].to_json(orient='records')
        # poo_docs = [data[i] for i in poo_docs]
        poo_docs = [index_to_id[i] for i in poo_docs]
        poo_docs = [products_info['shampoo'][id] for id in poo_docs]
        poo_matches = json.dumps(poo_docs)



        # poo_docs = tdidf_matrix[:cond_min_index]
        # poo_docs = cosine_similarity(poo_docs, query_vec).flatten() #shoud have len(poo_docs)
        # # poo_docs = np.argmax(poo_docs) #now just a single index
        # poo_docs = np.argsort(poo_docs)[:-6:-1] #top 5 in descending order: -1 to -5 inclusive
        # poo_matches = data.iloc[poo_docs].to_json(orient='records')

    if wantCond:
        bad_idx_cond = [x-cond_min_index for x in bad_idx if x<oil_min_index and x>=cond_min_index]
        cond_docs = docs_compressed_normalized[cond_min_index:oil_min_index]
        cond_docs = np.delete(cond_docs, bad_idx_cond, axis=0)
        cond_docs = np.dot(cond_docs, query_vec)
        cond_docs = np.argsort(-cond_docs)[:5] #sort descending, then get first 5
        # cond_matches = data.iloc[cond_docs].to_json(orient='records')
        cond_docs = [index_to_id[i + cond_min_index] for i in cond_docs]
        cond_docs = [products_info['conditioner'][id] for id in cond_docs]
        cond_matches = json.dumps(cond_docs)


        # cond_docs = tdidf_matrix[cond_min_index:oil_min_index]
        # cond_docs = cosine_similarity(cond_docs, query_vec).flatten()
        # cond_docs = np.argsort(cond_docs)[:-6:-1] #top 5 in descending order: -1 to -5 inclusive
        # cond_docs += cond_min_index
        # cond_matches = data.iloc[cond_docs].to_json(orient='records')
    if wantOil:
        bad_idx_oil = [x-oil_min_index for x in bad_idx if x>=oil_min_index]
        oil_docs = docs_compressed_normalized[oil_min_index:]
        oil_docs = np.delete(oil_docs, bad_idx_oil, axis=0)
        oil_docs = np.dot(oil_docs, query_vec)
        oil_docs = np.argsort(-oil_docs)[:5] #sort descending, then get first 5
        # oil_matches = data.iloc[oil_docs].to_json(orient='records')
        oil_docs = [index_to_id[i + oil_min_index] for i in oil_docs]
        oil_docs = [products_info['oil'][id] for id in oil_docs]
        oil_matches = json.dumps(oil_docs)


        # oil_docs = tdidf_matrix[oil_min_index:]
        # oil_docs = cosine_similarity(oil_docs, query_vec).flatten() 
        # oil_docs = np.argsort(oil_docs)[:-6:-1] #top 5 in descending order: -1 to -5 inclusive
        # oil_docs += oil_min_index
        # oil_matches = data.iloc[oil_docs].to_json(orient='records')
    

    #return should be dict or list of dicts;
    #JS will interpret at actual JSON
    # print('shampoo matches', poo_matches)
    # print()
    # print('conditioner matches', cond_matches)
    # print()
    # print('oil matches', oil_matches)
    return {'shampoos': poo_matches, 'conditioners': cond_matches, 'oils': oil_matches}


@app.route("/")
def home():
    # return render_template('base.html',title="sample html")
    return render_template('demo.html',title="demo html")

@app.route("/reviews")
def reviews_search():
    query = request.args.get("query")
    wantPoo = request.args.get("wantPoo") == 'true'
    wantCond = request.args.get("wantCond") == 'true'
    wantOil = request.args.get("wantOil") == 'true'

    bad_idx = set()
    with open(allergies_file_path, 'r') as f:
        allergies = json.load(f)
    if(request.args.get('no_sulfate') == 'true'):
        bad_idx.update(allergies['sulfates'])
    if(request.args.get('no_paraben') == 'true'):
        bad_idx.update(allergies['parabens'])
    if(request.args.get('no_fragrance') == 'true'):
        bad_idx.update(allergies['fragrances'])
    if(request.args.get('no_seed') == 'true'):
        bad_idx.update(allergies['seeds'])
    if(request.args.get('no_nut') == 'true'):
        bad_idx.update(allergies['nuts'])
    if(request.args.get('no_alcohol') == 'true'):
        bad_idx.update(allergies['alcohols'])
    
    bad_idx = sorted(list(bad_idx))


    return reviews_json_search(wantPoo, wantCond, wantOil, query, bad_idx)
    # return reviews_json_search(text)

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)