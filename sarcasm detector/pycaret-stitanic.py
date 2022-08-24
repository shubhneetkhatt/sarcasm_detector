#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#!pip install scikit-learn==0.23.2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




train_data = pd.read_csv("E:/flask_wapp/train.csv")
#train_data.dropna(inplace=True)
#print(train_data.head())
print(train_data.isna().sum())


# In[3]:


new_train = train_data["Cabin"].str.split("/",n=2,expand=True)

#print(new_train.head)

train_data["C1"]=new_train[0]
train_data["C2"]=pd.to_numeric(new_train[1])
train_data["C3"]=new_train[2]
train_data.drop(columns = ["Cabin"],inplace = True)



print(train_data.head())


# In[4]:


train_data.loc[train_data.CryoSleep == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0
train_data=train_data.fillna(train_data.median(numeric_only= True))
print(train_data.isna().sum())


# In[5]:


y = train_data["Transported"]

features = ["HomePlanet", "CryoSleep","Destination","Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck","C1","C2","C3" ]
X = pd.get_dummies(train_data[features], prefix=['HomePlanet', 'CryoSleep','Destination','VIP','C1','C3'], columns=['HomePlanet', 'CryoSleep','Destination','VIP','C1','C3'])

# In[6]:


from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=5, add_indicator=True)
knn.fit(X)
knn.transform(X)



print(X.isna().sum())
print(X.head())


# In[7]:


#import tensorflow as tf
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
print(scaler.fit(X))
#tf.convert_to_tensor(X)

#print(X)


# In[8]:


#!pip install imbalanced-learn==0.7.0 #--target=/kaggle/working/mysitepackages

#get_ipython().system('pip install pycaret[full] #--target=/kaggle/working/mysitepackages')
#!pip install pycaret[full]

# In[9]:


#X['Transported'] = y


# In[10]:



print(X.isna().sum())
print(X.head())


# In[11]:


#train_data.dropna(inplace=True)


# In[12]:


from pycaret.classification import *
numeric_cols = train_data.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(train_data.columns) - set(numeric_cols))
object_cols.remove("Transported")
ignore_cols = ["PassengerId","Name"]
clf = setup(data=train_data,
            target='Transported',
            normalize = True,
            normalize_method = 'robust',
            create_clusters = True,
            #feature_interaction = True,
            numeric_features = numeric_cols,
            categorical_features = object_cols,
            ignore_features = ignore_cols,
            session_id = 42,
            use_gpu = False,
            silent = True,
            fold = 5,
            train_size=0.99,
            n_jobs = -1)


# In[13]:


top = compare_models(sort = 'Accuracy', n_select = 8)


# In[14]:


stack = blend_models(top, optimize='Accuracy')
#stack = ensemble_model(top, method = 'Bagging')
predict_model(stack)
final_stack = finalize_model(stack)

plot_model(final_stack, plot = 'confusion_matrix')


# In[15]:


import gc
gc.collect()
#predictions = predict_model(final_stack, data=test_data)


# In[16]:


print(predictions.head)

