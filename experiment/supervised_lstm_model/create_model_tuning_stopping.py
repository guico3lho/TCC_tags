#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


from gensim.models import Word2Vec
import json
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import seaborn as sns
import keras_tuner as kt
from keras.optimizers import SGD

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

import matplotlib.pyplot as plt

from keras.preprocessing.text import tokenizer_from_json
import os,sys


# In[2]:


print(sys.executable)
print(os.getcwd())


# In[3]:


# with open('../../assets/word_index.json', 'r') as f:
#     word_index = json.load(f)
#     word_index = dict(word_index)


# In[4]:


with open('../../assets/deep_assets/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    word_index = tokenizer.word_index


# In[5]:


train = pd.read_csv('../../assets/data/splits/train/padded.csv')
val = pd.read_csv('../../assets/data/splits/val/padded.csv')

X_train = train.to_numpy()[:, :-1]
y_train = train.to_numpy()[:, -1]

X_val = val.to_numpy()[:, :-1]
y_val = val.to_numpy()[:, -1]


# In[6]:


model_we = Word2Vec.load('../../assets/deep_assets/word2vec.model')

model_we.wv.most_similar('petrobras')


# In[7]:


# List of nparrays of size 300
embeddings_dict = {}
for word in model_we.wv.index_to_key:
    embeddings_dict[word] = model_we.wv[word]


# In[8]:


# create matrix with vocab train words
embeddings_on_this_context = np.zeros((len(word_index), 300))
for word, i in word_index.items():
    embeddings_vector = embeddings_dict.get(word)
    if embeddings_vector is not None:
        embeddings_on_this_context[i - 1] = embeddings_vector


# ## Functions

# In[9]:


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


# ## RNN Model

# ### Without Tuning and Early stopping

# In[10]:


# model = Sequential([
#     Embedding(input_dim=len(word_index), output_dim= 300, input_length=X_train.shape[1], trainable=False, weights=[embeddings_on_this_context]),
#     Bidirectional(LSTM(64, return_sequences=True)),
#     # Dropout(0.4),
#     # Bidirectional(LSTM(hp.Choice('units',[32,64]))),
#     Bidirectional(LSTM(64)),
#     Dense(32, activation='relu'),
#     # Dropout(0.6),
#     Dense(4, activation='softmax')
# ])
#
# model.summary()

# from keras.optimizers import SGD
#
# loss = "sparse_categorical_crossentropy"
# optimizer = SGD(learning_rate=0.01)
# metrics = ['accuracy']
#
# model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
# history = model.fit(X_train, y_train, epochs=25, validation_data=(X_val,y_val), verbose=2)

# history = model.fit(X_train, y_train, epochs=4000, validation_data=(X_val,y_val), verbose=2, callbacks=[es])


# ## With Tuning and Early Stopping

# In[13]:


def build_model(hp):
    model = Sequential([
    Embedding(input_dim=len(word_index), output_dim= 300, input_length=X_train.shape[1], trainable=False, weights=[embeddings_on_this_context]),
    Bidirectional(LSTM(hp.Choice('units_bilstm_1',[16,32,64]), return_sequences=True)),
    Dropout(hp.Float('rate_dp_1',0.5,0.9,step=0.1,default=0.5)),
    Bidirectional(LSTM(hp.Choice('units_bilstm_2',[16,32,64]))),
    Dense(hp.Choice('units_dense',[16,32,64]), hp.Choice('activation',['relu','sigmoid','tanh']) ),
    Dropout(hp.Float('rate_dp_2',0.5,0.9,step=0.1,default=0.5)),
    Dense(4, activation='softmax')
])
    loss = "sparse_categorical_crossentropy"
    # optimizer = SGD(learning_rate=0.01)
    metrics = ['accuracy']

    model.compile(loss=loss,optimizer=hp.Choice('optimizer',['adam','sgd','rmsprop']),metrics=metrics)
    return model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# In[14]:


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=4,
    executions_per_trial=1,
    directory='../../assets/deep_assets',
    project_name='lstm_tuning')

tuner.search_space_summary()


# In[15]:


tuner.search(X_train, y_train, epochs=4000, validation_data=(X_val,y_val), callbacks=[es])


# In[ ]:


tuner.results_summary()


# In[ ]:


best_model = tuner.get_best_models()[0]


# ### Evaluation

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# ### Exporting model

# In[ ]:


model.save('../../assets/deep_assets/lstm_model')

