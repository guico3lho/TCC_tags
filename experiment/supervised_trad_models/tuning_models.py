#!/usr/bin/env python
# coding: utf-8

# ## Packages

# In[1]:


import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit



# ## Classes and Functions

# ## Dependencies

# In[2]:


train = pd.read_csv('../../assets/data/splits/train/preprocessed.csv')
val = pd.read_csv('../../assets/data/splits/val/preprocessed.csv')


# In[3]:


y_train = train['label']
y_val = val['label']


# In[4]:


with open('../../assets/trad_assets/cv_vec.pkl', 'rb') as fout:
    cv_vec = pickle.load(fout)


# In[5]:


with open('../../assets/trad_assets/tfidf_vec.pkl', 'rb') as fout:
    tfidf_vec = pickle.load(fout)


# ## Grid Params

# ### KNN

# In[7]:


knn_params = {
    # 'n_neighbors': [11, 21, 40, 60, 80, 100],
    'n_neighbors': [15 ,17 ,19, 21, 23, 25, 27],
    'metric': ['cosine'],
    'weights': ['distance']
    # 'weights': ['uniform', 'distance'],
    # 'metric': ['cosine']
}


# 

# ### SVM

# In[8]:


svm_params = {
    # 'C': [1,10, 50, 100],
    'C': [1,5,10],
    # 'C': [1,5,10],
    # 'kernel': ['rbf']
    # 'kernel' : ['rbf'],
}


# ### Naive Bayes

# In[9]:


nb_params = {
    'alpha': [0.1, 1, 10],
    'fit_prior': [True, False]
}


# ### Logistic Regression

# 

# In[10]:


lr_params = {
    'penalty': ['l1','l2', None],
    'C': [0.1, 1, 10],
    'solver': ['liblinear','sag', 'saga']
}


# ## Tuning

# In[11]:


split_index = [-1] * len(train) + [0] * len(val)

X = pd.concat([train, val], axis=0, ignore_index=True)

y = np.concatenate((y_train, y_val), axis=0)
pds = PredefinedSplit(test_fold=split_index)





# ### Count Vectorizer

# In[12]:


X_cv = cv_vec.transform(X['title']).toarray()
pd.DataFrame(X_cv, columns=cv_vec.get_feature_names_out())


# In[12]:





# In[13]:


# from hypopt import GridSearch

model_params = ([KNeighborsClassifier(), SVC(), MultinomialNB(), LogisticRegression()],
                [knn_params, svm_params, nb_params, lr_params])

list_best_models_params = []
for model, params in zip(model_params[0], model_params[1]):
    gs = GridSearchCV(model,
                      param_grid=params,
                      )

    gs.fit(X_cv, y)
    print(f"Best CV results for {model.__class__.__name__}")
    print("Best Score of train set: " + str(gs.best_score_))
    print("Best estimator: " + str(gs.best_estimator_))
    print("Best parameter set: " + str(gs.best_params_))

    store_best_model_configs = {
        'model_name': model.__class__.__name__,
        'best_score': gs.best_score_,
        'best_estimator': gs.best_estimator_,
        'best_params': gs.best_params_
    }

    list_best_models_params.append(store_best_model_configs)

df_best_models_params = pd.DataFrame(list_best_models_params)
df_best_models_params.to_csv('../../assets/trad_assets/best_models_params_cv.csv', index=False)

df_best_models_params
# cv_best_model = gs.best_estimator_
# print("Test Score: " + str(gs.score(X_val_cv, y_val)))
# print("----------------------------------------------------")


# In[14]:


cv_best_model = gs.best_estimator_
cv_best_model


# ### TF-IDF

# In[15]:


# X_train_tfidf = tfidf_vec.transform(X_train['title'])
# X_val_tfidf = tfidf_vec.transform(X_val['title'])
# X_train_tfidf
X_tfidf = tfidf_vec.transform(X['title']).toarray()
pd.DataFrame(X_tfidf, columns=tfidf_vec.get_feature_names_out())


# In[16]:


model_params = ([KNeighborsClassifier(), SVC(), MultinomialNB(), LogisticRegression()],
                [knn_params, svm_params, nb_params, lr_params])

list_best_models_params = []
for model, params in zip(model_params[0], model_params[1]):
    gs = GridSearchCV(model,
                      param_grid=params,
                      )
    gs.fit(X_tfidf, y)
    print(f"Best TF-IDF results for {model.__class__.__name__}")
    print("Best Score on train set: " + str(gs.best_score_))
    print("Best estimator: " + str(gs.best_estimator_))
    print("Best parameter set: " + str(gs.best_params_) + "\n")
    store_best_model_configs = {
        'model_name': model.__class__.__name__,
        'best_score': gs.best_score_,
        'best_estimator': gs.best_estimator_,
        'best_params': gs.best_params_
    }

    list_best_models_params.append(store_best_model_configs)

df_best_models_params = pd.DataFrame(list_best_models_params)
df_best_models_params.to_csv('../../assets/trad_assets/best_models_params_tfidf.csv', index=False)
df_best_models_params


# decide_best_model =
# print("Test Score: " + str(gs.score(X_val, y_val)))
# print("----------------------------------------------------")


# ## Best Models for each type

# In[1]:


import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit



# ### Dependencies

# In[2]:


train = pd.read_csv('../../assets/data/splits/train/preprocessed.csv')
val = pd.read_csv('../../assets/data/splits/val/preprocessed.csv')


# In[3]:


y_train = train['label']
y_val = val['label']


# In[4]:


with open('../../assets/trad_assets/cv_vec.pkl', 'rb') as fout:
    cv_vec = pickle.load(fout)


# In[5]:


with open('../../assets/trad_assets/tfidf_vec.pkl', 'rb') as fout:
    tfidf_vec = pickle.load(fout)


# ### Models

# In[6]:


X_cv = cv_vec.transform(train['title']).toarray()
# pd.DataFrame(X_cv, columns=cv_vec.get_feature_names_out())
cv_best_model_knn = KNeighborsClassifier(n_neighbors=21, weights='distance', metric='cosine').fit(X_cv, train['label'])
cv_best_model_svm = SVC(C=10, kernel='rbf').fit(X_cv, train['label'])
cv_best_model_nb = MultinomialNB(alpha=1, fit_prior=True).fit(X_cv, train['label'])
cv_best_model_lr = LogisticRegression(C=1, solver='liblinear', penalty='l2').fit(X_cv, train['label'])
cv_best_models = [cv_best_model_knn, cv_best_model_svm, cv_best_model_nb, cv_best_model_lr]
cv_best_models


# In[7]:


X_tfidf = tfidf_vec.transform(train['title']).toarray()
# pd.DataFrame(X_cv, columns=tfidf.get_feature_names_out())
tfidf_best_model_knn = KNeighborsClassifier(metric='cosine', n_neighbors=60, weights='distance').fit(X_tfidf, train['label'])
tfidf_best_model_svm = SVC(C=10, kernel='rbf').fit(X_tfidf, train['label'])
tfidf_best_model_nb = MultinomialNB(alpha=10, fit_prior=False).fit(X_tfidf, train['label'])
tfidf_best_model_lr = LogisticRegression(C=10, solver='liblinear', penalty='l2').fit(X_tfidf, train['label'])
tfidf_best_models = [tfidf_best_model_knn, tfidf_best_model_svm, tfidf_best_model_nb, tfidf_best_model_lr]
tfidf_best_models


# ## Outputs

# In[8]:


with open('../../assets/trad_assets/cv_set.pkl', 'wb') as fout:
    pickle.dump((cv_vec, cv_best_models), fout)

with open('../../assets/trad_assets/tfidf_set.pkl', 'wb') as fout:
    pickle.dump((tfidf_vec, tfidf_best_models), fout)

