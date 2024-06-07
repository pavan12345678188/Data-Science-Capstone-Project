#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv('CAR DETAILS.csv')
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df.columns


# In[10]:


df.duplicated().sum()


# In[21]:


df.drop_duplicates()


# In[22]:


df.duplicated().sum()


# In[23]:


df.duplicated()


# In[36]:


df.duplicated().sum()


# In[38]:


df.describe()


# In[42]:


categorical_cols=df.select_dtypes(include=['object']).columns
categorical_cols


# In[43]:


numerical_cols=df.select_dtypes(include=['int64','float64']).columns
numerical_cols


# In[44]:


df['name'].value_counts()


# In[45]:


df['year'].value_counts()


# In[46]:


df['selling_price'].value_counts()


# In[47]:


df['seller_type'].value_counts()


# In[48]:


df['km_driven'].value_counts()


# In[49]:


df['fuel'].value_counts()


# In[50]:


df['transmission'].value_counts()


# In[52]:


df['owner'].value_counts()


# In[ ]:




