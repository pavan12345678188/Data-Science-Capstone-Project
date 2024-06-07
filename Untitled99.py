#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv('CAR DETAILS.csv')
df


# In[3]:


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoder.fit(df)
df_encoded=encoder.transform(df)
df_encoded


# In[4]:


df['selling_price'].mean()


# In[5]:


df['selling_price'].mean()


# In[6]:


df['year'].mean()


# In[7]:


df['name'].mode()


# In[8]:


df['year'].mode()


# In[9]:


df['selling_price'].mode()


# In[10]:


df['km_driven'].mode()


# In[11]:


df['fuel'].mode()


# In[12]:


df['seller_type'].mode()


# In[13]:


df['transmission'].mode()


# In[14]:


df['owner'].mode()


# In[15]:


df['selling_price'].median()


# In[16]:


df['selling_price'].median()


# In[17]:


df['year'].median()


# In[20]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols
# Initialize the scaler
scaler = StandardScaler()
# Scale numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first few rows of the cleaned and pre-processed dataset
print(df.head())


# In[22]:


from sklearn.model_selection import train_test_split

# Define feature columns (X) and target column (y)
X = df.drop(columns=['name'])
y = df['name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[ ]:




