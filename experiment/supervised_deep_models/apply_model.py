#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


import keras.models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import seaborn as sns
import json
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.text import tokenizer_from_json

from keras.utils import pad_sequences
import pandas as pd
import re
import string
import nltk


# In[2]:


model = keras.models.load_model('../../assets/deep_assets/lstm_model')
model.summary()


# In[3]:


# Load the tokenizer from the file
with open('../../assets/deep_assets/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    word_index = tokenizer.word_index


# 

# ## Functions

# In[4]:


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


# ## Loading test data

# In[5]:


test_raw = pd.read_csv('../../assets/data/splits/test/raw.csv')
test_raw


# In[6]:


test_preprocessed = transformDocument(test_raw, 'title', 'pt')

X_test = test_preprocessed.title
y_test = test_preprocessed.to_numpy()[:, -1]

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=11, padding='post', truncating='post')
test_padded


# ### Functions

# In[7]:


def index2word(word_index):
    index_word = {}
    for key in word_index:
        index_word[word_index[key]] = key
    return index_word


def seq2text(seq, index_word):
    text = []
    for index in seq:
        text.append(index_word[index])
    return text

def show_confusion_matrix(cm):
        print("Confusion Matrix")
        plt.figure(figsize=(10, 7))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('Confusion Matrix')
        plt.show()


# In[8]:


index_word = index2word(word_index)


# In[9]:


pred_prob = model.predict(test_padded)


# In[10]:


# y_pred = [1 if p > 0.5 else 0 for p in pred_prob]
y_pred = np.argmax(pred_prob, axis=1)


# In[11]:


df_results = pd.DataFrame()
X_test = list(X_test)
y_test = list(y_test)
df_results['sequence'] = test_sequences
df_results['X_test'] = X_test
df_results['seq2text'] = df_results['sequence'].apply(lambda x: seq2text(x, index_word))
df_results['y_pred'] = y_pred
df_results['y_true'] = y_test
df_results


# ### Metrics

# In[12]:


# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# precision = precision_score(average='macro', y_true=y_test, y_pred=y_pred)
# recall = recall_score(average='macro', y_true=y_test, y_pred=y_pred)
# f1 = f1_score(average='macro', y_true=y_test, y_pred=y_pred)
# cm = confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[13]:


print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
show_confusion_matrix(cm)

