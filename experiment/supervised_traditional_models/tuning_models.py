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


# ## Tuning Models

# ### Logistic Regression

# In[6]:


# parameter grid
parameters = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}

logreg = LogisticRegression(max_iter=10000, multi_class='ovr')
clf = model_selection.GridSearchCV(logreg,  # model
                                   param_grid=parameters,  # hyperparameters
                                   scoring='accuracy',  # metric for scoring
                                   cv=10)

clf.fit(X_train, y_train)
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :", clf.best_score_)


# ### Naive Bayes

# In[ ]:


# gridsearch for Naive Bayes
parameters = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'fit_prior': [True, False]
}

nb = MultinomialNB()
clf = model_selection.GridSearchCV(nb,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10)
clf.fit(X_train, y_train)
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :", clf.best_score_)


# ### KNN

# In[ ]:


# gridsearch for KNN
parameters = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

knn = KNeighborsClassifier()
clf = model_selection.GridSearchCV(knn,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10)

clf.fit(X_train, y_train)
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :", clf.best_score_)


# ### SVM

# In[ ]:


# parameters = [
#     {'kernel': ['linear', 'poly']}
# ]
# svm = SVC(C=1)
# clf = model_selection.GridSearchCV(svm,
#                                    param_grid=parameters,
#                                    scoring='accuracy',
#                                    cv=10)
#
# clf.fit(X_train, y_train)
# print("Tuned Hyperparameters :", clf.best_params_)
# print("Accuracy :", clf.best_score_)

