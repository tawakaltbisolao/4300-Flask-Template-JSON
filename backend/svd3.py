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

# cond_min_index = 298
# oil_min_index = 540

data_path = 'dataset/ulta_reviews.json'
vocab_file = 'dataset/ulta_vocabulary.json'
with open(data_path) as f:
    data = json.load(f)

vec = TfidfVectorizer(max_df=0.4, min_df=2, ngram_range=(1,2))
tfidf = vec.fit_transform(data)
with open(vocab_file, 'w') as f:
    json.dump(vec.vocabulary_, f)

docs_compressed, s, words_compressed = svds(tfidf, k=40)
words_compressed = words_compressed.transpose()

docs_compressed = normalize(docs_compressed, axis=1)
words_compressed = normalize(words_compressed, axis=1)
# print(words_compressed.shape)

with open('svd_NEWDATA_U.npy', 'wb') as f:
    np.save(f, docs_compressed)
with open('svd_NEWDATA_V001.npy', 'wb') as f:
    np.save(f, words_compressed[:200_000])
with open('svd_NEWDATA_V002.npy', 'wb') as f:
    np.save(f, words_compressed[200_000:])
