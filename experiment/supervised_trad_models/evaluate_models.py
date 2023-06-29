#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import pickle


# ## Dependencies

# In[2]:


test = pd.read_csv('../../assets/data/splits/test/preprocessed.csv')
y_test = test['label']


# In[3]:


with open('../../assets/trad_assets/cv_set.pkl', 'rb') as fout:
    cv_vec, cv_best_models = pickle.load(fout)

with open('../../assets/trad_assets/tfidf_set.pkl', 'rb') as fout:
    tfidf_vec, tfidf_best_models = pickle.load(fout)


# In[4]:


cv_best_models


# In[5]:


tfidf_best_models


# ## Functions

# In[6]:


def viewPredictedRows(X_test, y_test, y_pred):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df['correct'] = df['y_test'] == df['y_pred']
    df['correct'] = df['correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    df['title'] = X_test
    return df


# In[7]:


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


def show_graph_metrics(y_test, y_pred, modelo):
    df_metrics = pd.DataFrame(index=['Outros', 'Petrobras', 'Vale', 'Itaú'])
    df_metrics['Acurácia'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    df_metrics['Precisão'] = precision_score(y_true=y_test, y_pred=y_pred, average=None)
    df_metrics['Recall'] = recall_score(y_true=y_test, y_pred=y_pred, average=None)
    df_metrics['F1-Score'] = f1_score(y_true=y_test, y_pred=y_pred, average=None)
    fig = px.bar(df_metrics, height=500, width=750,  x=df_metrics.index, y=["Acurácia", "Precisão", "Recall", "F1-Score"],
             barmode="group", title=f"Desempenho de {modelo} em relação a precisão, recall e F1-Score", labels={'index': 'Classes', 'value': 'Porcentagem (%)', 'variable': 'Métricas'})

    fig.show()


# ## Predictions considering best traditional models for both vectorization

# ### CountVectorizer

# In[9]:


X_test = cv_vec.transform(test['title']).toarray()
X_test_names = pd.DataFrame(X_test, columns=cv_vec.get_feature_names_out())
X_test_names


# In[10]:


print("CountVectorizer models")
for model in cv_best_models:
    print("Model: ", model.__class__.__name__)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    show_graph_metrics(y_test, y_pred, model.__class__.__name__)




# In[11]:


# df_results = viewPredictedRows(test['title'], y_test, y_pred)
# df_results
# cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
# show_confusion_matrix(cm)


# ### TfidfVectorizer

# In[12]:


X_test = tfidf_vec.transform(test['title']).toarray()
X_test_names = pd.DataFrame(X_test, columns=tfidf_vec.get_feature_names_out())
X_test_names


# In[13]:


print("TFIDF models")
for model in tfidf_best_models:
    print("Model: ", model.__class__.__name__)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    show_graph_metrics(y_test, y_pred, model.__class__.__name__)

