#!/usr/bin/env python
# coding: utf-8

# 1) Delivery_time -> Predict delivery time using sorting time

# In[41]:


import numpy as np
import pandas as pd


# In[42]:


df=pd.read_csv('delivery_time.csv')
df


# In[46]:


X=df[['Sorting Time']]
Y=df['Delivery Time']

import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


# In[47]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)


# In[48]:


LR.intercept_


# In[49]:


LR.coef_


# In[50]:


Y_pred=LR.predict(X)


# In[51]:


import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X,Y_pred,color='black')
plt.show()


# In[52]:


from sklearn.metrics import mean_squared_error


# In[53]:


mse=mean_squared_error(Y,Y_pred)
print('Mean Squared Error:',mse.round(2))


# In[54]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error:',rmse.round(2))


# Hence, Simple Regression Model for prediction of delivery time using sorting time is prepared.

# In[ ]:





# 2) Salary_hike -> Build a prediction model for Salary_hike
# 

# In[55]:


df=pd.read_csv('Salary_Data.csv')
df


# In[61]:


Y=df['Salary']
X=df[['YearsExperience']]


# In[62]:


import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


LR=LinearRegression()
LR.fit(X,Y)


# In[65]:


LR.intercept_


# In[66]:


LR.coef_


# In[67]:


Y_pred=LR.predict(X)


# In[72]:


plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X,Y_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[69]:


from sklearn.metrics import mean_squared_error


# In[70]:


mse=mean_squared_error(Y,Y_pred)
print('Mean Squared Error:',mse.round(2))


# In[71]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error:',rmse.round(2))


# Hence, Simple Regression Model for prediction of Salary using YearsExperience is prepared.

# In[ ]:




