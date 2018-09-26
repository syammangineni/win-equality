
# coding: utf-8

# In[1]:


from google.colab import files
upload = file.upload()


# In[3]:


import pandas as pd
dataframe= pd.read_csv('winequality-red.csv')


# In[4]:


dataframe.info()


# In[5]:


dataframe.describe()


# In[6]:


#data visualization:
import seaborn as sns


# In[7]:


sns.pairplot(data = dataframe)


# In[105]:


#FEATURES
features = dataframe[['fixed acidity','volatile acidity','critric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','ph','sulphates','alcohol']]


# In[8]:


#target = dataframe['grade']
target = dataframe['density']


# In[9]:


#normalization
import matplotlib.pyplot as plt
plt.hist(target,bins=10)


# In[108]:


#solving normalization
import numpy as np


# In[109]:


#log trasformation
dataframe_norm = np.log(target)


# In[110]:


plt.hist(dataframe_norm)


# In[111]:


#dont apply normalization
#train test split
from sklearn.model_selection import train_test_split


# In[10]:


#result with out normalization
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,train_size=0.8)


# In[11]:


#performance matric
from sklearn.metrics import r2_score


# In[12]:


#linear regression model
from sklearn.linear_model import LinearRegression


# In[13]:


#Feeding the Model
regressor = LinearRegression()
reg_fit = regressor.fit(x_train,y_train)
reg_pred = reg_fit.predict(x_test)


# In[ ]:


#score 
score_not_norm = r2_score(y_test,reg_pred)


# In[ ]:


score_not_norm


# In[ ]:


#result with normalization
x_train_n,x_test_n,y_train_n,y_test_n = train_test_split(features,dataframe_norm,test_size=0.2,train_size=0.8)


# In[123]:


#feeding the model
regressor_n = LinearRegression()
reg_fit_n = regressor_n.fit(x_train_n,y_train_n)
reg_pred_n = reg_fit_n.predict(x_test_n)


# In[124]:


score_norm = r2_score(y_test_n,reg_pred_n)


# In[125]:


print(score_norm)


# In[27]:




