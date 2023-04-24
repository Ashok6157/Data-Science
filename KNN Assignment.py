#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a model for glass classification using KNN
# 
# Data Description:
# 
# RI : refractive index
# 
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 
# Mg: Magnesium
# 
# AI: Aluminum
# 
# Si: Silicon
# 
# K:Potassium
# 
# Ca: Calcium
# 
# Ba: Barium
# 
# Fe: Iron
# 
# Type: Type of glass: (class attribute)
# 1 -- building_windows_float_processed
#  2 --building_windows_non_float_processed
#  3 --vehicle_windows_float_processed
#  4 --vehicle_windows_non_float_processed (none in this database)
#  5 --containers
#  6 --tableware
#  7 --headlamps
# 

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('glass.csv')
df


# In[4]:


Y=df['Type']
X=df.iloc[:,0:9]
list(X)


# In[6]:


from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X=pd.DataFrame(SS_X)
SS_X.columns=list(X)
SS_X


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(SS_X,Y,train_size=0.7)


# In[8]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[10]:


from sklearn.metrics import accuracy_score


# In[11]:


print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:





# 2) Implement a KNN model to classify the animals in to categorie

# In[12]:


import pandas as pd


# In[13]:


df=pd.read_csv('Zoo.csv')
df


# In[26]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=LabelEncoder()
df['animal name']=LE.fit_transform(df['animal name'])
df['animal name']


# In[27]:


Y=df['type']
X=df.iloc[:,0:17]
list(X)


# In[28]:


SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X=pd.DataFrame(SS_X)
SS_X.columns=list(X)
SS_X


# In[29]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(SS_X,Y,train_size=0.7)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:




