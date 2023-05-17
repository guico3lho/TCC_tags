#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


import pandas as pd
import re
import string
import nltk
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from gensim.models import Word2Vec


# In[2]:


df_pp = pd.read_csv('../../assets/data/df_pp.csv')
df_pp


# ## Functions

# ## Loading Word2vec Model

# ## Word2Vec Model

# In[3]:


hasModel = False
if not hasModel:

    tokenized_corpus = []
    for title in df_pp['title']:
        tokenized_corpus.append(title.split())
    model = Word2Vec(sentences=tokenized_corpus, vector_size=300, window=20, min_count=5,workers =4)
    model.save("../../assets/deep_assets/word2vec.model")
else:
    print("Model Exists!")
    model = Word2Vec.load("../../assets/deep_assets/word2vec.model")
    print(model.wv.most_similar('queda'))


# In[4]:


model.wv.get_normed_vectors()


# In[5]:


model.wv.index_to_key


# In[6]:


word_embeddings = []
vocab = []

for i, word in enumerate(model.wv.index_to_key):
    vocab.append(word)
    word_embeddings.append(model.wv[word])


# In[7]:


emb_df = pd.DataFrame(word_embeddings)
emb_df = emb_df.assign(nome=vocab)
emb_df


# In[8]:


tsne = TSNE(n_components=2,perplexity=40, n_iter=2500, random_state=23, learning_rate=100,metric='euclidean',init='pca')
X_tsne = tsne.fit_transform(np.array(word_embeddings))
X_tsne[1:4,:]
emb_df['x'] = X_tsne[:,0]
emb_df['y'] = X_tsne[:,1]
fig = px.scatter(emb_df,x='x',y='y',hover_data={'x':False, 'y':False,'nome':True})
fig.show()


# 
