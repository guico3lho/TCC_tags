#!/usr/bin/env python
# coding: utf-8

# ## Packages and Assets

# In[1]:


from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df_pp = pd.read_csv('../assets/data/df_pp.csv')

# remover numeros dos titulos
# df_eda = df_pp.copy()
# df_eda['title'] = df_eda['title'].str.replace(r'\d+', '')
# df_eda
df_pp


# In[3]:


petr4 = df_pp.loc[df_pp['label'] == 0]
vale3 = df_pp.loc[df_pp['label'] == 1]
itub4 = df_pp.loc[df_pp['label'] == 2]


# ## Functions

# In[4]:


stopwords = ["da", "meu", "em", "voce", "de", "ao", "os", "<NUM>", "<num>", "num", "NUM", 'acoe', 'acoes', 'milhoe',' bilhoe', 'milhoes', 'cento', 'diz', 'bilhoes']


# In[5]:


summary_petr4 = " ".join(str(s) for s in petr4['title'].values)
summary_vale3 = " ".join(str(s) for s in vale3['title'].values)
summary_itub4 = " ".join(str(s) for s in itub4['title'].values)


# ## PETR4

# ### Wordcloud

# 

# In[6]:


wordcloud_petr4 = WordCloud(collocations=False, stopwords=stopwords, background_color='white', width=1600, height=800).generate(summary_petr4)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_petr4)
plt.axis('off')
plt.title("Words related to PETR4")
plt.show()


# ### See news that doesnt have the word "petrobras" or "petr4"

# In[7]:


petr4


# In[8]:


petr4.loc[(petr4['title'].str.contains('petrobras') == False) & (petr4['title'].str.contains('petr4') == False)]


# ## VALE3

# ### WordCloud

# In[9]:


wordcloud_vale3 = WordCloud(collocations=False, stopwords=stopwords, background_color='white', width=1600, height=800).generate(summary_vale3)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_vale3)
plt.axis('off')
plt.title("Words related to VALE3")
plt.show()


# ### See news that doesnt have the word "vale" or "vale3"

# In[10]:


vale3


# In[11]:


vale3.loc[(vale3['title'].str.contains('vale3') == False) & (vale3['title'].str.contains('vale') == False)]


# ## ITUB4

# ### WordCloud

# In[12]:


wordcloud_itub4 = WordCloud(collocations=False, stopwords=stopwords, background_color='white', width=1600, height=800).generate(summary_itub4)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_itub4)
plt.axis('off')
plt.title("Words related to ITU4")
plt.show()


# ### See news that doesnt have the word "itau" or "itub4"

# In[13]:


itub4


# In[14]:


itub4.loc[(itub4['title'].str.contains('itau') == False) & (itub4['title'].str.contains('itub4') == False)]


# In[14]:





# In[14]:




