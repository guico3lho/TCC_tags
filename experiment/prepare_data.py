#!/usr/bin/env python
# coding: utf-8

# ## Packages

# In[2]:


import json
import pandas as pd
import os
import sys
import requests
# import gdown
import opendatasets as od

print(sys.executable)
print(os.getcwd())

# 0 - petr4
# 1 - vale3
# 2 - itub4
# df_vale.explode('tags').groupby('tags').count().sort_values('title', ascending=False)


# ## Load data

# ### [1] PETR4

# #### Suno

# In[4]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/suno/suno-petr4.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()
df_suno_petr_raw = pd.DataFrame(data)
df_suno_petr = df_suno_petr_raw[
    df_suno_petr_raw['tags'].apply(lambda x: 'Vale (VALE3)' not in x and 'Itaú Unibanco (ITUB3/ITUB4)' not in x)]
df_suno_petr = df_suno_petr.rename(columns={'url': 'link'})
df_suno_petr


# In[5]:


df_suno_petr.explode('tags').groupby('tags').count().sort_values('title', ascending=False)


# #### Moneytimes

# In[6]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/moneytimes/moneytimes-petr4.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_moneytimes_petr_raw = pd.DataFrame(data)

df_moneytimes_petr = df_moneytimes_petr_raw[
    df_moneytimes_petr_raw['tags'].apply(lambda x: 'Vale' not in x and 'Itaú Unibanco' not in x)]
df_moneytimes_petr


# #### Concat suno with moneytimes

# In[ ]:





# In[8]:


df_petr4 = pd.concat([df_suno_petr[['title', 'tags', 'link']], df_moneytimes_petr[['title', 'tags', 'link']]])

df_petr4.reset_index(inplace=True, drop=True)

df_petr4['title'] = df_petr4['title'].map(lambda s: s.replace('\xa0', ''))

df_petr4['label'] = 1

df_petr4


# ### [2] VALE3

# #### Suno

# In[9]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/suno/suno-vale3.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_suno_vale_raw = pd.DataFrame(data)

df_suno_vale = df_suno_vale_raw[df_suno_vale_raw['tags'].apply(lambda
                                                                   x: 'Petrobras (PETR4)' not in x and 'PETR3' not in x and 'Itaú Unibanco (ITUB3/ITUB4)' not in x)]
df_suno_vale = df_suno_vale.rename(columns={'url': 'link'})
df_suno_vale


# #### Moneytimes

# In[10]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/moneytimes/moneytimes-vale3.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_moneytimes_vale_raw = pd.DataFrame(data)

df_moneytimes_vale = df_moneytimes_vale_raw[
    df_moneytimes_vale_raw['tags'].apply(lambda x: 'Petrobras' not in x and 'Itaú Unibanco' not in x)]
df_moneytimes_vale


# #### Concat suno with moneytimes

# In[11]:


df_vale3 = pd.concat([df_suno_vale[['title', 'tags', 'link']], df_moneytimes_vale[['title', 'tags', 'link']]])

df_vale3.reset_index(inplace=True, drop=True)

df_vale3['title'] = df_vale3['title'].map(lambda s: s.replace('\xa0', ''))

df_vale3['label'] = 2

df_vale3


# ### [3] ITUB4

# #### Suno

# In[12]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/suno/suno-itub4.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_suno_itub4_raw = pd.DataFrame(data)

df_suno_itub4 = df_suno_itub4_raw[df_suno_itub4_raw['tags'].apply(
    lambda x: 'Petrobras (PETR4)' not in x and 'PETR3' not in x and 'Vale (VALE3)' not in x)]
df_suno_itub4 = df_suno_itub4.rename(columns={'url': 'link'})
df_suno_itub4


# #### Moneytimes

# In[14]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/moneytimes/moneytimes-itub4.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_moneytimes_itub4_raw = pd.DataFrame(data)

df_moneytimes_itub4 = df_moneytimes_itub4_raw[
    df_moneytimes_itub4_raw['tags'].apply(lambda x: 'Petrobras' not in x and 'Vale' not in x)]
df_moneytimes_itub4


# #### Infomoney

# In[16]:


json_data_path = 'https://raw.githubusercontent.com/guico3lho/DataScience_Assets/main/Datasets/financial_market/infomoney_test/infomoney-results.json'

# url = 'https://raw.githubusercontent.com/guico3lho/TCC/main/assets/Datasets/suno/suno-petr4.json?token=GHSAT0AAAAAABYDRQ4ZASOZJOMO2OAKI4MEZAOJOKA'
r = requests.get(json_data_path)
data = r.json()

df_infomoney_raw = pd.DataFrame(data)

df_infomoney_itub4 = df_infomoney_raw[df_infomoney_raw['tags'].apply(lambda x: 'Itaú' in x)]

# df_moneytimes_vale = df_moneytimes_vale_raw[
#     df_moneytimes_vale_raw['tags'].apply(lambda x: 'Petrobras' not in x and 'Itaú Unibanco' not in x)]
# df_moneytimes_vale


# In[17]:


df_itub4 = pd.concat([df_suno_itub4[['title', 'tags', 'link']], df_moneytimes_itub4[['title', 'tags', 'link']], df_infomoney_itub4[['title','tags','link']]])

df_itub4.reset_index(inplace=True, drop=True)

df_itub4['title'] = df_itub4['title'].map(lambda s: s.replace('\xa0', ''))

df_itub4['label'] = 3

df_itub4


# ### [0] Notícias do SUNO que não são PETR4, ITUB4 ou VALE3 using Kaggle

# #### Importing

# In[18]:


# https://drive.google.com/file/d/1OGmCmxSVM0SFdbce6zRXQ458xGmUo5Xs/view?usp=sharing

od.download('https://www.kaggle.com/datasets/guico3lho/suno-news-2018-2020', '../assets/data')


# In[19]:


json_data_path = '../assets/data/suno-news-2018-2020/results-full-suno-2020.json'

with open(json_data_path, 'r', encoding='utf8') as json_file:
    data = json.load(json_file)

df_suno_raw = pd.DataFrame(data)
# df_suno_raw


# #### Filters to remove stocks that are PETR4, ITUB4 or VALE3

# In[20]:


df_suno_other_stocks = df_suno_raw[df_suno_raw['tags'].apply(lambda x: 'PETR4' not in x
                                                                       and 'PETR3' not in x
                                                                       and 'Petrobras' not in x
                                                                       and 'Vale (VALE3)' not in x
                                                                       and 'Vale' not in x
                                                                       and 'VALE3' not in x
                                                                       and 'Itaú Unibanco (ITUB4)' not in x
                                                                       and 'Itau Unibanco' not in x
                                                                       and 'ITUB4' not in x
                                                                       and 'ITUB3' not in x
                                                                       and 'Itau' not in x

                                                             )]



# In[21]:


# df_suno_raw[df_suno_raw['tags'].apply(lambda x: lambda y: y.str.contains('Petrobras') for y in x)]
# df_suno_raw[df_suno_raw['tags'].apply(lambda x: lambda y: y.str.contains('Vale') for y in x)]

# df_suno_other_stocks[df_suno_other_stocks['tags'].apply(lambda x: 'Itau' in x)]


# In[22]:


df_other_stocks = df_suno_other_stocks.sample(6000)


# In[23]:


df_other_stocks = df_other_stocks[['title', 'tags', 'url']]
df_other_stocks = df_other_stocks.rename(columns={'url': 'link'})
df_other_stocks['label'] = 0
df_other_stocks.reset_index(inplace=True, drop=True)


# ## Concat all loaded data and shuffle to generate final dataframe

# In[24]:


df_concat = pd.concat([df_petr4, df_vale3, df_itub4, df_other_stocks])
df = df_concat.sample(frac=1).reset_index(drop=True)
df


# In[25]:


df.value_counts('label')


# ## Split data into train (80%), val (10%) and test (10%) sets

# In[26]:


from sklearn.model_selection import train_test_split

# Split data into train and test
train, rem = train_test_split(df, train_size=0.8, random_state=52)  # 80% train

val, test = train_test_split(rem, test_size=0.5, random_state=52)  # 10% val, 10% testc


# ## Data Augmentation test

# In[27]:


data_to_augmentate = train.loc[train['label'] == 3]


# In[28]:


sample_title = data_to_augmentate.iloc[2]['title']
sample_title


# In[29]:


import translators as ts


# In[30]:


sample_title_en = ts.translate_text(sample_title, from_language='pt', to_language='en')
sample_title_en


# 

# In[31]:


sample_title_pt = ts.translate_text(sample_title_en, from_language='en', to_language='pt')
sample_title_pt


# ## Split size

# In[35]:


df.value_counts('label')


# In[32]:


print(train.value_counts('label'))
print(val.value_counts('label'))
print(test.value_counts('label'))


# In[33]:


train.value_counts('label').plot(kind='bar')


# ## Export full df, train, val and test for future use

# In[34]:


df.to_csv('../assets/data/df.csv', index=False)
train.to_csv('../assets/data/splits/train/raw.csv', index=False)
val.to_csv('../assets/data/splits/val/raw.csv', index=False)
test.to_csv('../assets/data/splits/test/raw.csv', index=False)

