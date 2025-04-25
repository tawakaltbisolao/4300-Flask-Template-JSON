import json
import os
import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib
import matplotlib.pyplot as plt

products_path = 'dataset/products_ulta.json'
shampoo_path = 'dataset/ulta_shampoo_reviews.json'
conditioner_path = 'dataset/ulta_conditioner_reviews.json'
oil_path = 'dataset/ulta_oil_reviews.json'

with open(products_path, 'r') as f:
    products_info = json.load(f)

shampoo_info = products_info['shampoo']
conditioner_info = products_info['conditioner']
oil_info = products_info['oil']
# keys are product IDs, 
# keys for each product entry are 'product', 'brand', 'rating', 'price', 'values', 'ingredients', 'url', 'imgUrl'

cond_min_index = len(shampoo_info)
oil_min_index = cond_min_index + len(conditioner_info)

productIds = list(shampoo_info.keys()) + list(conditioner_info.keys()) + list(oil_info.keys())
#list of product IDs
id_to_index = {val: i for i, val in enumerate(productIds)}

concatenatedReviews = [''] * len(productIds)
reviewCounts = dict()

with open(shampoo_path, 'r') as f:
    reviews = json.load(f) #will reuse this var
    #also prepend ingredient lists
for r_dict in reviews:
    #keys are 'product_id', 'headline', 'content', 'helpfulness', 'stars'
    index = id_to_index[r_dict['product_id']]
    if(reviewCounts.get(index, 0) >= 1000):
        continue
    # concatenatedReviews[index] += r_dict['headline'] + ' ' + r_dict['content']
    concatenatedReviews[index] += r_dict['content']
    reviewCounts[index] = reviewCounts.get(index, 0) + 1

with open(conditioner_path, 'r') as f:
    reviews = json.load(f) 
for r_dict in reviews:
    index = id_to_index[r_dict['product_id']]
    if(reviewCounts.get(index, 0) >= 1000):
        continue
    # concatenatedReviews[index] += r_dict['headline'] + ' ' + r_dict['content']
    concatenatedReviews[index] += r_dict['content']
    reviewCounts[index] = reviewCounts.get(index, 0) + 1
    
with open(oil_path, 'r') as f:
    reviews = json.load(f) 
for r_dict in reviews:
    index = id_to_index[r_dict['product_id']]
    if(reviewCounts.get(index, 0) >= 1000):
        continue
    # concatenatedReviews[index] += r_dict['headline'] + ' ' + r_dict['content']
    concatenatedReviews[index] += r_dict['content']
    reviewCounts[index] = reviewCounts.get(index, 0) + 1

reviews = None

with open('dataset/ulta_product_ids.json', 'w') as f:
    json.dump(productIds, f)
    #setting in stone the order of productIds to use


for id in productIds:
    index = id_to_index[id]
    ingredients = ''
    if index < cond_min_index:
        for ing in shampoo_info[id]:
            # ingredients += ing.strip('.') + ' , '
            ingredients += ' , ' + ing.strip('.')
    elif index < oil_min_index:
        for ing in conditioner_info[id]:
            # ingredients += ing.strip('.') + ' , '
            ingredients += ' , ' + ing.strip('.')
    else:
        for ing in oil_info[id]:
            # ingredients += ing.strip('.') + ' , '
            ingredients += ' , ' + ing.strip('.')
    concatenatedReviews[index] += ingredients #+ concatenatedReviews[index]

with open('dataset/ulta_reviews.json', 'w') as f:
    json.dump(concatenatedReviews, f)


#now time to vectorize + SVD concatenatedReviews
# vec = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2))
# tfidf = vec.fit_transform(concatenatedReviews)

