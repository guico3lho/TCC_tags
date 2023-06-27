#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import json


# In[2]:


train = pd.read_csv('../../assets/data/splits/train/preprocessed.csv')
val = pd.read_csv('../../assets/data/splits/val/preprocessed.csv')


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[4]:


y_train = train['label']
cv = CountVectorizer(ngram_range=(1, 1))
X_train = cv.fit_transform(train['title']).toarray()
X_train_names = pd.DataFrame(X_train, columns=cv.get_feature_names_out())
X_train_names


# In[5]:


X_val = cv.transform(val['title']).toarray()
y_val = val['label']


# ## Functions

# In[6]:


def evaluateModels(X_train, y_train, models, n_splits):
    print(f"{n_splits}-Fold Cross validation")
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f"{name}: Mean Accuracy={cv_results.mean():.5f}, Standard Deviation={cv_results.std():.5f}")


# In[7]:


def viewPredictedRows(X_test, y_test, y_pred):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df['correct'] = df['y_test'] == df['y_pred']
    df['correct'] = df['correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    df['title'] = X_test
    return df


# In[8]:


def evaluateModelsWithoutKfold(X_train, y_train, X_test, y_test, models):
    results = []
    names = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append(accuracy)
        names.append(name)
        print(f"{name}: Accuracy={accuracy:.5f}")


# ## Evaluating Models

# In[9]:


models = []

models.append(('LR', LogisticRegression(max_iter=10000, multi_class='ovr', C=0.001, penalty='l2', solver='newton-cg')))
models.append(('SVM', SVC(C=1, kernel='linear')))
models.append(('KNN', KNeighborsClassifier(metric='cosine', n_neighbors=7, weights='distance')))
models.append(('NB', MultinomialNB(alpha=0.1, fit_prior=True)))

evaluateModelsWithoutKfold(X_train, y_train, X_val, y_val, models)


# In[10]:


# models = []
#
# models.append(('LR', LogisticRegression(max_iter=10000, multi_class='ovr', C=0.001, penalty='l2', solver='newton-cg')))
# models.append(('SVM', SVC(C=1, kernel='linear')))
# models.append(('KNN', KNeighborsClassifier(metric='cosine', n_neighbors=7, weights='distance')))
# models.append(('NB', MultinomialNB(alpha=0.1, fit_prior=True)))
#
# evaluateModelsWithoutKfold(X_train, y_train, X_val, y_val, models)


# In[11]:


from sklearn.metrics import classification_report

model = MultinomialNB(alpha=0.1, fit_prior=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))


# In[12]:


df_results = viewPredictedRows(val['title'], y_val, y_pred)
df_results


# ## Exports

# In[13]:


with open('../../assets/trad_assets/count_vectorizer_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(cv.vocabulary_, f)


# In[14]:


import pickle
filename = 'naive_bayes_model.sav'
pickle.dump(model, open(f"../../assets/trad_assets/{filename}", 'wb'))


# In[15]:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score


# In[16]:


# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                            alpha=1e-3, random_state=42,
#                                            max_iter=5, tol=None)),
#                      ])
#


# In[17]:


# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (1e-2, 1e-3),
# }


# In[18]:


# gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)


# In[19]:


# gs_clf = gs_clf.fit(train['text'], train['label'])


# In[19]:





# In[19]:





# 
