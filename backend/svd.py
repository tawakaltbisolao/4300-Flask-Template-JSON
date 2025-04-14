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

cond_min_index = 35501
oil_min_index = 67892


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))


json_file_path = os.path.join(current_directory, 'init.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
data = [review['profile'] + ' ' + review['content'] for review in data]
vec = TfidfVectorizer(max_df=0.8, min_df=10, ngram_range=(1,3))
tfidf = vec.fit_transform(data)

docs_compressed,s,words_compressed = svds(tfidf, k=30)
words_compressed = words_compressed.transpose()

docs_compressed = normalize(docs_compressed, axis=1)
words_compressed = normalize(words_compressed, axis=1)

with open('svd.npy', 'wb') as f:
    np.save(f, docs_compressed)
    np.save(f, words_compressed)


# word_to_index = vec.vocabulary_
# index_to_word = {i:t for t,i in word_to_index.items()}



# print('U shape: ' + str(u.shape))
# print('S shape: ' + str(s.shape))
# print('V.T shape: ' + str(v_trans.shape))


# plt.plot(s[::-1])
# plt.xlabel("Singular value number")
# plt.ylabel("Singular value")
# plt.show()