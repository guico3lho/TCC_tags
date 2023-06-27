#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from gensim.models import Word2Vec
import numpy as np
import json


# In[2]:


df_train_preprocessed = pd.read_csv('../../assets/data/splits/train/preprocessed.csv')
df_val_preprocessed = pd.read_csv('../../assets/data/splits/val/preprocessed.csv')
df_test_preprocessed = pd.read_csv('../../assets/data/splits/test/preprocessed.csv')


# ## Functions

# In[3]:


def createVocabulary(corpus):
    """
    - Cria vocab (palavra e sua respectiva frequência no corpus)
    - Cria tokens (palavras do corpus)

    :param corpus: lista de string
    :return vocab, tokens, vocab_size:
    """
    tokens = []  # {'deeds', 'old', ...} 71666
    vocab = {}  # {'deeds': 2, 'old': 20', ...} 17971
    for text in corpus:
        for token in text.split():
            tokens.append(token)
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_size = len(vocab)
    return tokens, vocab, vocab_size


# In[4]:


def findMaxLen(sequence):
    max_len = 0
    for text in sequence:
        if len(text) > max_len:
            max_len = len(text)
    return max_len

def findAverageLen(sequence):
    total_len = 0
    for text in sequence:
        total_len += len(text)
    return total_len / len(sequence)


# ## Separating each split into features (X) and target (y)

# In[5]:


X_train = df_train_preprocessed.title
y_train = df_train_preprocessed.label

X_val = df_val_preprocessed.title
y_val = df_val_preprocessed.label


# ## Generating tokens and vocabulary

# In[6]:


tokens, vocab, vocab_size = createVocabulary(X_train)
len(vocab)


# ## Numericalization

# In[7]:


tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(X_train)


# ## Padding

# In[8]:


max_len = findMaxLen(train_sequences)
max_len = int(max_len/2)


# In[9]:


train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
train_padded = np.insert(train_padded, 11, y_train, 1)

val_sequences = tokenizer.texts_to_sequences(X_val)
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding='post', truncating='post')
val_padded = np.insert(val_padded, 11, y_val, 1)


# ## Exports

# In[10]:


# Convert the tokenizer to a dictionary
tokenizer_json = tokenizer.to_json()

# Save the tokenizer to a file
with open('../../assets/deep_assets/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)


# In[11]:


pd.DataFrame(train_padded).to_csv('../../assets/data/splits/train/padded.csv', index=False)
pd.DataFrame(val_padded).to_csv('../../assets/data/splits/val/padded.csv', index=False)


# In[11]:




