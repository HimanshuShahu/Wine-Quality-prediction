#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


wine_dataset = pd.read_csv(r"E:\Wine Quality prediction\winequality-red.csv")


# In[3]:


wine_dataset.shape


# In[4]:


wine_dataset.head()


# In[5]:


wine_dataset.isnull().sum()


# In[6]:


wine_dataset.describe()


# In[7]:


sns.catplot(x='quality',data=wine_dataset, kind='count')


# In[8]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)


# In[9]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = wine_dataset)


# In[10]:


correlation = wine_dataset.corr()


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')


# In[12]:


X = wine_dataset.drop('quality',axis=1)


# In[13]:


print(X)


# In[14]:


Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[15]:


print(Y)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[17]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[18]:


model = RandomForestClassifier()


# In[19]:


model.fit(X_train, Y_train)


# In[20]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[21]:


print('Accuracy : ', test_data_accuracy)


# In[22]:


input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')

