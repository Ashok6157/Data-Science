#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.

# In[35]:


import numpy as np
import pandas as pd


# In[36]:


df=pd.read_csv('50_Startups.csv')
df


# In[3]:


Y=df['Profit']
print(df.corr())


# In[37]:


X1=df[['R&D Spend']]
X2=df[['R&D Spend','Administration']]
X3=df[['R&D Spend','Administration','Marketing Spend']]


# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


LR=LinearRegression()
LR.fit(X1,Y)


# In[40]:


LR.intercept_


# In[41]:


LR.coef_


# In[42]:


Y_pred1=LR.predict(X1)


# In[43]:


import matplotlib.pyplot as plt
plt.scatter(X1,Y)
plt.scatter(X1,Y_pred1,color='red')
plt.plot(X1,Y_pred1,color='black')
plt.show()


# In[44]:


from sklearn.metrics import mean_squared_error,r2_score


# In[45]:


mse=mean_squared_error(Y,Y_pred1)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred1)
print('R2 score is',r2.round(2))


# In[46]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[47]:


LR=LinearRegression()
LR.fit(X2,Y)


# In[48]:


LR.intercept_


# In[49]:


LR.coef_


# In[50]:


Y_pred2=LR.predict(X2)


# In[51]:


mse=mean_squared_error(Y,Y_pred2)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred2)
print('R2 score is',r2.round(2))


# In[52]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[53]:


LR=LinearRegression()
LR.fit(X3,Y)


# In[54]:


LR.intercept_


# In[55]:


LR.coef_


# In[56]:


Y_pred3=LR.predict(X3)


# In[57]:


mse=mean_squared_error(Y,Y_pred3)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred3)
print('R2 score is',r2.round(2))


# In[58]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[59]:


X4=df[['R&D Spend','Marketing Spend']]


# In[60]:


LR=LinearRegression()
LR.fit(X4,Y)


# In[61]:


LR.intercept_


# In[62]:


LR.coef_


# In[63]:


Y_pred4=LR.predict(X4)


# In[64]:


mse=mean_squared_error(Y,Y_pred4)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred4)
print('R2 score is',r2.round(2))


# In[32]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[ ]:





# 2) Consider only the below columns and prepare a prediction model for predicting Price.
# 
# Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

# In[70]:


df = pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
df


# In[89]:


Y=df['Price']
X=df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


# In[93]:


df1=df[['Price',"Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
df1


# In[94]:


df1.corr()


# Strong +ve =(KM,Age_08_04),(Weight,Quarterly_Tax) 

# In[95]:


X1=df[["Age_08_04","HP"]]
X2=df[["Age_08_04","cc"]]
X3=df[["Age_08_04","Doors"]]
X4=df[['Age_08_04','Gears']]
X5=df[['Age_08_04','Quarterly_Tax']]
X6=df[['Age_08_04','Weight']]
X7=df[['KM','HP']]
X8=df[['KM','HP']]
X9=df[['KM','cc']]
X10=df[['KM','Doors']]
X11=df[['KM','Gears']]
X12=df[['KM','Quarterly_Tax']]
X13=df[['KM','Weight']]
X14=df[['HP','cc']]
X15=df[['HP','Doors']]
X16=df[['HP','Gears']]
X17=df[['HP','Quarterly_Tax']]
X18=df[['HP','Weight']]
X19=df[['cc','Doors']]
X20=df[['cc','Gears']]
X21=df[['cc','Quarterly_Tax']]
X22=df[['cc','Weight']]
X23=df[['Doors','Gears']]
X24=df[['Doors','Quarterly_Tax']]
X25=df[['Doors','Weight']]
X26=df[['Gears','Quarterly_Tax']]
X27=df[['Gears','Weight']]
X28=df[['Age_08_04','HP','cc']]
X29=df[['HP','cc','Doors']]
X30=df[['Age_08_04','HP','cc','Doors']]
X31=df[['HP','Weight','KM','cc']]


# In[96]:


from sklearn.linear_model import LinearRegression


# In[97]:


LR=LinearRegression()
LR.fit(X1,Y)


# In[98]:


LR.intercept_


# In[99]:


LR.coef_


# In[100]:


Y_pred1=LR.predict(X1)


# In[101]:


from sklearn.metrics import mean_squared_error,r2_score


# In[102]:


mse=mean_squared_error(Y,Y_pred1)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred1)
print('R2 score is',r2.round(2))


# In[103]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[104]:


LR=LinearRegression()
LR.fit(X2,Y)


# In[105]:


LR.intercept_


# In[106]:


LR.coef_


# In[108]:


Y_pred2=LR.predict(X2)


# In[109]:


mse=mean_squared_error(Y,Y_pred2)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred2)
print('R2 score is',r2.round(2))


# In[110]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[111]:


LR=LinearRegression()
LR.fit(X3,Y)


# In[112]:


LR.intercept_


# In[113]:


LR.coef_


# In[114]:


Y_pred3=LR.predict(X3)


# In[115]:


mse=mean_squared_error(Y,Y_pred3)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred3)
print('R2 score is',r2.round(2))


# In[116]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[117]:


LR=LinearRegression()
LR.fit(X4,Y)


# In[118]:


LR.intercept_


# In[119]:


LR.coef_


# In[120]:


Y_pred4=LR.predict(X4)


# In[121]:


mse=mean_squared_error(Y,Y_pred4)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred4)
print('R2 score is',r2.round(2))


# In[122]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[123]:


LR=LinearRegression()
LR.fit(X5,Y)


# In[124]:


LR.intercept_


# In[125]:


LR.coef_


# In[126]:


Y_pred5=LR.predict(X5)


# In[127]:


mse=mean_squared_error(Y,Y_pred5)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred5)
print('R2 score is',r2.round(2))


# In[128]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[129]:


LR=LinearRegression()
LR.fit(X6,Y)


# In[130]:


LR.intercept_


# In[131]:


LR.coef_


# In[132]:


Y_pred6=LR.predict(X6)


# In[133]:


mse=mean_squared_error(Y,Y_pred6)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred6)
print('R2 score is',r2.round(2))


# In[134]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[135]:


LR=LinearRegression()
LR.fit(X7,Y)


# In[136]:


LR.intercept_


# In[137]:


LR.coef_


# In[138]:


Y_pred7=LR.predict(X7)


# In[139]:


mse=mean_squared_error(Y,Y_pred7)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred7)
print('R2 score is',r2.round(2))


# In[140]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[141]:


LR=LinearRegression()
LR.fit(X8,Y)


# In[142]:


LR.intercept_


# In[143]:


LR.coef_


# In[144]:


Y_pred8=LR.predict(X8)


# In[145]:


mse=mean_squared_error(Y,Y_pred8)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred8)
print('R2 score is',r2.round(2))


# In[146]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[147]:


LR=LinearRegression()
LR.fit(X9,Y)


# In[148]:


LR.intercept_


# In[149]:


LR.coef_


# In[150]:


Y_pred9=LR.predict(X9)


# In[151]:


mse=mean_squared_error(Y,Y_pred9)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred9)
print('R2 score is',r2.round(2))


# In[152]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[153]:


LR=LinearRegression()
LR.fit(X10,Y)


# In[154]:


LR.intercept_


# In[155]:


LR.coef_


# In[156]:


Y_pred10=LR.predict(X10)


# In[157]:


mse=mean_squared_error(Y,Y_pred10)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred10)
print('R2 score is',r2.round(2))


# In[158]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[159]:


LR=LinearRegression()
LR.fit(X11,Y)


# In[160]:


LR.intercept_


# In[161]:


LR.coef_


# In[162]:


Y_pred11=LR.predict(X11)


# In[163]:


mse=mean_squared_error(Y,Y_pred11)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred11)
print('R2 score is',r2.round(2))


# In[164]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[165]:


LR=LinearRegression()
LR.fit(X12,Y)


# In[166]:


LR.intercept_


# In[167]:


LR.coef_


# In[168]:


Y_pred12=LR.predict(X12)


# In[169]:


mse=mean_squared_error(Y,Y_pred12)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred12)
print('R2 score is',r2.round(2))


# In[170]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[171]:


LR=LinearRegression()
LR.fit(X13,Y)


# In[172]:


LR.intercept_


# In[173]:


LR.coef_


# In[174]:


Y_pred13=LR.predict(X13)


# In[175]:


mse=mean_squared_error(Y,Y_pred13)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred13)
print('R2 score is',r2.round(2))


# In[176]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[177]:


LR=LinearRegression()
LR.fit(X14,Y)


# In[178]:


LR.intercept_


# In[179]:


LR.coef_


# In[180]:


Y_pred14=LR.predict(X14)


# In[181]:


mse=mean_squared_error(Y,Y_pred14)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred14)
print('R2 score is',r2.round(2))


# In[182]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[183]:


LR=LinearRegression()
LR.fit(X15,Y)


# In[184]:


LR.intercept_


# In[185]:


LR.coef_


# In[186]:


Y_pred15=LR.predict(X15)


# In[187]:


mse=mean_squared_error(Y,Y_pred15)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred15)
print('R2 score is',r2.round(2))


# In[188]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[189]:


LR=LinearRegression()
LR.fit(X16,Y)


# In[190]:


LR.intercept_


# In[191]:


LR.coef_


# In[192]:


Y_pred16=LR.predict(X16)


# In[193]:


mse=mean_squared_error(Y,Y_pred16)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred16)
print('R2 score is',r2.round(2))


# In[194]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[195]:


LR=LinearRegression()
LR.fit(X17,Y)


# In[196]:


LR.intercept_


# In[197]:


LR.coef_


# In[198]:


Y_pred17=LR.predict(X17)


# In[199]:


mse=mean_squared_error(Y,Y_pred17)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred17)
print('R2 score is',r2.round(2))


# In[200]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[201]:


LR=LinearRegression()
LR.fit(X18,Y)


# In[202]:


LR.intercept_


# In[203]:


LR.coef_


# In[204]:


Y_pred18=LR.predict(X18)


# In[205]:


mse=mean_squared_error(Y,Y_pred18)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred18)
print('R2 score is',r2.round(2))


# In[206]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[207]:


LR=LinearRegression()
LR.fit(X19,Y)


# In[208]:


LR.intercept_


# In[209]:


LR.coef_


# In[210]:


Y_pred19=LR.predict(X19)


# In[211]:


mse=mean_squared_error(Y,Y_pred19)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred19)
print('R2 score is',r2.round(2))


# In[212]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[213]:


LR=LinearRegression()
LR.fit(X20,Y)


# In[214]:


LR.intercept_


# In[215]:


LR.coef_


# In[216]:


Y_pred20=LR.predict(X20)


# In[217]:


mse=mean_squared_error(Y,Y_pred20)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred20)
print('R2 score is',r2.round(2))


# In[218]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[219]:


LR=LinearRegression()
LR.fit(X21,Y)


# In[220]:


LR.intercept_


# In[221]:


LR.coef_


# In[222]:


Y_pred21=LR.predict(X21)


# In[223]:


mse=mean_squared_error(Y,Y_pred21)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred21)
print('R2 score is',r2.round(2))


# In[224]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[225]:


LR=LinearRegression()
LR.fit(X22,Y)


# In[226]:


LR.intercept_


# In[227]:


LR.coef_


# In[228]:


Y_pred22=LR.predict(X22)


# In[229]:


mse=mean_squared_error(Y,Y_pred22)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred22)
print('R2 score is',r2.round(2))


# In[230]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[231]:


LR=LinearRegression()
LR.fit(X23,Y)


# In[232]:


LR.intercept_


# In[233]:


LR.coef_


# In[234]:


Y_pred23=LR.predict(X23)


# In[235]:


mse=mean_squared_error(Y,Y_pred23)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred23)
print('R2 score is',r2.round(2))


# In[236]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[237]:


LR=LinearRegression()
LR.fit(X24,Y)


# In[238]:


LR.intercept_


# In[239]:


LR.coef_


# In[240]:


Y_pred24=LR.predict(X24)


# In[241]:


mse=mean_squared_error(Y,Y_pred24)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred24)
print('R2 score is',r2.round(2))


# In[242]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[243]:


LR=LinearRegression()
LR.fit(X25,Y)


# In[244]:


LR.intercept_


# In[245]:


LR.coef_


# In[246]:


Y_pred25=LR.predict(X25)


# In[247]:


mse=mean_squared_error(Y,Y_pred25)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred25)
print('R2 score is',r2.round(2))


# In[248]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[249]:


LR=LinearRegression()
LR.fit(X26,Y)


# In[250]:


LR.intercept_


# In[251]:


LR.coef_


# In[252]:


Y_pred26=LR.predict(X26)


# In[253]:


mse=mean_squared_error(Y,Y_pred26)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred26)
print('R2 score is',r2.round(2))


# In[254]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[255]:


LR=LinearRegression()
LR.fit(X27,Y)


# In[256]:


LR.intercept_


# In[257]:


LR.coef_


# In[258]:


Y_pred27=LR.predict(X27)


# In[259]:


mse=mean_squared_error(Y,Y_pred27)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred27)
print('R2 score is',r2.round(2))


# In[260]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[261]:


LR=LinearRegression()
LR.fit(X28,Y)


# In[262]:


LR.intercept_


# In[263]:


LR.coef_


# In[264]:


Y_pred28=LR.predict(X28)


# In[265]:


mse=mean_squared_error(Y,Y_pred28)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred28)
print('R2 score is',r2.round(2))


# In[266]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[267]:


LR=LinearRegression()
LR.fit(X29,Y)


# In[268]:


LR.intercept_


# In[269]:


LR.coef_


# In[270]:


Y_pred29=LR.predict(X29)


# In[271]:


mse=mean_squared_error(Y,Y_pred29)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred29)
print('R2 score is',r2.round(2))


# In[272]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[273]:


LR=LinearRegression()
LR.fit(X30,Y)


# In[274]:


LR.intercept_


# In[275]:


LR.coef_


# In[276]:


Y_pred30=LR.predict(X30)


# In[277]:


mse=mean_squared_error(Y,Y_pred30)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred30)
print('R2 score is',r2.round(2))


# In[278]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[279]:


LR=LinearRegression()
LR.fit(X31,Y)


# In[280]:


LR.intercept_


# In[281]:


LR.coef_


# In[282]:


Y_pred31=LR.predict(X31)


# In[283]:


mse=mean_squared_error(Y,Y_pred31)
print('Mean Squared Error is',mse.round(2))
r2=r2_score(Y,Y_pred31)
print('R2 score is',r2.round(2))


# In[284]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error is',rmse.round(2))


# In[ ]:




