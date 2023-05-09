#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('book.csv',encoding ='iso-8859-1')
df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.sort_values('User.ID')
len(df)


# In[7]:


len(df['User.ID'].unique())


# In[8]:


len(df['Book.Title'].unique())


# In[9]:


df['Book.Rating'].value_counts()


# In[10]:


df['Book.Rating'].hist()


# In[11]:


user_df1 = df.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')
user_df1


# In[12]:


user_df1.fillna(0, inplace=True)
user_df1


# In[13]:


from sklearn.metrics import pairwise_distances
user_sim1 = 1 - pairwise_distances(user_df1,metric='cosine')


# In[14]:


user_sim1.shape


# In[16]:


user_sim_df1 = pd.DataFrame(user_sim1)
user_sim_df1


# In[17]:


user_sim_df1.index   = df['User.ID'].unique()
user_sim_df1.columns = df['User.ID'].unique()


# In[18]:


user_sim_df1.head()


# In[19]:


user_sim_df1.shape


# In[23]:


np.fill_diagonal(user_sim1, 0)


# In[24]:


user_sim_df1.idxmax(axis=1)[0:10]


# In[25]:


df[(df['User.ID']==276729) | (df['User.ID']==276726)]


# In[ ]:





# In[26]:


user_df2 = df.pivot_table(index='Book.Title',columns='User.ID',values='Book.Rating')
user_df2


# In[27]:


user_df2.fillna(0, inplace=True)
user_df2


# In[28]:


from sklearn.metrics import pairwise_distances
user_sim2 = 1 - pairwise_distances(user_df2,metric='cosine')


# In[29]:


user_sim2.shape


# In[30]:


user_sim_df2 = pd.DataFrame(user_sim2)
user_sim_df2


# In[32]:


user_sim_df2.index   = df['Book.Title'].unique()
user_sim_df2.columns = df['Book.Title'].unique()


# In[33]:


user_sim_df2.head()


# In[34]:


user_sim_df2.shape


# In[35]:


np.fill_diagonal(user_sim2, 0)


# In[39]:


user_sim_df2.idxmax(axis=1)[0:10]


# In[41]:


df[(df['Book.Title']=='The Mummies of Urumchi') | (df['Book.Title']=='My First Cousin Once Remo')]


# In[ ]:




