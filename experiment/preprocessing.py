#!/usr/bin/env python
# coding: utf-8

# ## Packages

# In[15]:


import pandas as pd
import re
import string
import nltk
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# In[ ]:


jupyter nbconvert --to script experiment/*.ipynb;
jupyter nbconvert --to script experiment/supervised_lstm_model/*.ipynb;
jupyter nbconvert --to script experiment/supervised_traditional_models/*.ipynb


rm experiment/*.ipynb;
rm experiment/supervised_lstm_model/*.ipynb;
rm experiment/supervised_traditional_models/*.ipynb;


# ## Functions

# In[16]:


def transformDocument(df, column_name, language):
    stop_words = usingStopwords(language)
    df_pp = df.copy()
    # 1. Aplicar preprocessamento nos títulos e textos completos
    if language == 'pt':
        # Substituir símbolos importantes
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('-feira', ''))
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('+', 'mais '))
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('-', 'menos '))
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('%', ' por cento'))
        df_pp[column_name] = df_pp[column_name].map(lambda s: removeStopwords(s, stop_words))

    elif language == 'en':
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('-', 'less'))
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('+', 'plus '))
        df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('%', ' percent'))
        df_pp[column_name] = df_pp[column_name].map(lambda s: removeStopwords(s, stop_words))

    else:
        pass

    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('R$', ''))
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('U$', ''))
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('US$', ''))
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('S&P 500', 'spx'))

    # Transformar em String e Letras Minúsculas nas Mensagens
    df_pp[column_name] = df_pp[column_name].map(lambda s:
                                              normalizarString(s))

    # Remover Pontuações
    # Remover Pontuações
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.translate(str.maketrans('', '', string.punctuation)))

    # Remover Emojis
    df_pp[column_name] = df_pp[column_name].map(lambda s: removeEmojis(s))

    # Quebras de Linha desnecessárias
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('\n', ' '))

    # Remover aspas duplas
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('\"', ''))
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('“', ''))
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.replace('”', ''))

    # Remover valores
    df_pp[column_name] = df_pp[column_name].map(lambda s: removeValores(s))

    # Espaços desnecessários
    df_pp[column_name] = df_pp[column_name].map(lambda s: s.strip())
    return df_pp



def removeEmojis(sentence):
    "Remoção de Emojis nas mensagens de texto."

    # Padrões dos Emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', sentence)

def removeValores(sentence):
    new_sentece = ''

    for token in sentence.split():
        if token.isdigit():
            token = '<NUM>'
        new_sentece += ' {}'.format(token)

    return new_sentece

def usingStopwords(language):
    stop_words = []

    nltk.download('stopwords')

    if language == 'pt':
        stop_words = nltk.corpus.stopwords.words('portuguese')
    elif language == 'en':
        stop_words = nltk.corpus.stopwords.words('english')

    return stop_words

def removeStopwords(text, stop_words):
    tokens = []
    for word in text.split():
        if word not in stop_words:
            tokens.append(word)

    text = ' '.join(tokens)
    return text

def normalizarString(text):
    """
    Função para retirar acentuações e converter para minúscula
    :param text:
    :return text_normalizado
    """
    import unicodedata

    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text.lower())


# ## Load splits

# In[17]:


train_raw = pd.read_csv('../assets/data/splits/train/raw.csv')
val_raw = pd.read_csv('../assets/data/splits/val/raw.csv')
test_raw = pd.read_csv('../assets/data/splits/test/raw.csv')
df = pd.read_csv('../assets/data/df.csv')


# In[18]:


df


# ## Preprocessing splits

# In[19]:


train_pp = transformDocument(train_raw, 'title', 'pt')
val_pp = transformDocument(val_raw, 'title', 'pt')
test_pp = transformDocument(test_raw, 'title', 'pt')
df_pp = transformDocument(df, 'title', 'pt')


# In[20]:


df_pp


# ## Exporting preprocessed splits for EDA, word2vec and lstm_preprocessing

# In[21]:


train_pp.to_csv('../assets/data/splits/train/preprocessed.csv', index=False)
val_pp.to_csv('../assets/data/splits/val/preprocessed.csv', index=False)
test_pp.to_csv('../assets/data/splits/test/preprocessed.csv', index=False)
df_pp.to_csv('../assets/data/df_pp.csv', index=False)

