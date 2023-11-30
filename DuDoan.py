#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# In[42]:


app = Flask(__name__)


# Data Collection and Processing

# In[4]:


df = pd.read_csv(r'D:\KhoaLuan\covid_19_indias.csv')


# In[5]:


df.head()


# In[6]:


print(df)


# In[8]:


df.shape


# In[7]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df['Correlate'].value_counts()


# In[12]:


df['Date'].value_counts()


# 

# In[13]:


mean_value = df.groupby('Correlate')['Correlate'].mean()
print(mean_value)


# In[14]:


print(df)


# In[15]:


df = df.drop(columns=['Date','Time','ConfirmedIndianNational','ConfirmedForeignNational'])


# In[16]:


print(df)


# In[17]:


X = df.drop(columns=['State/UnionTerritory','Correlate'], axis=1)
Y = df['Correlate']


# In[18]:


print(X)


# In[19]:


print(Y)


# Splitting the data to training data & Test data (Tách dữ liệu thành dữ liệu huấn luyện & dữ liệu kiểm tra)

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


column_names = X.columns.tolist()


# In[24]:


def example_route():
    state_territory = request.form.get('State/UnionTerritory')

# Chuẩn hóa dữ liệu

# In[25]:


scaler = StandardScaler()


# In[26]:


scaler.fit(X_train)


# In[27]:


X_train = pd.DataFrame(scaler.transform(X_train), columns=column_names)

X_test = pd.DataFrame(scaler.transform(X_test), columns=column_names)


# In[29]:


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[30]:


print(X_train)


# So sánh sự tương quan của các cột

# In[31]:


numeric_columns = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(18, 15))
sns.heatmap(numeric_columns.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
plt.title('Sự tương quan giữa các cột')
plt.show()


# In[32]:


sns.set()
fig = df.hist(figsize=(13,12), color='lightblue', xlabelsize=12, ylabelsize=12)
[x.title.set_size(8) for x in fig.ravel()]
plt.show()


# In[33]:


# xây dựng mô hình dự doán
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# mô hình 1: Regression
lg = LinearRegression()
lg.fit(X_train, Y_train)
lg_pred = lg.predict(X_test)
print("mô hình hồi quy: ",lg_pred)


# In[34]:


# Mô hình 2: SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)
svm_pred = svm_classifier.predict(X_test)
print("mô hình SVM:",svm_pred)


# In[35]:


# Mô hình : Random Forest
model = RandomForestClassifier()
model.fit(X_train, Y_train)
rfc_pred = model.predict(X_test)
print("mô hình random forest",rfc_pred)


# In[36]:


# Mô hình 4: binary tree
from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, Y_train)
bt_pred = tree_classifier.predict(X_test)
print("mô hình binary tree:",bt_pred)


# Đánh giá mô hình

# In[37]:


#đánh giá độ chính xác của từng mô hình
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
lg_accuracy = mean_squared_error(Y_test, lg_pred)
rfc_accuracy = accuracy_score(Y_test, rfc_pred)
svm_accuracy = accuracy_score(Y_test, svm_pred)
bt_accuracy = accuracy_score(Y_test, bt_pred)
print('Độ chính xác của mô hình hồi quy: ',lg_accuracy)
print('Độ chính xác của mô hình Random Forest: ',rfc_accuracy)
print('Độ chính xác của mô hình SVM : ',svm_accuracy)
print('Độ chính xác của mô hình binary tree : ',bt_accuracy)


# In[38]:


# Mô hình : Random Forest
model = RandomForestClassifier()
model.fit(X_train, Y_train)

import joblib
joblib.dump(model, 'your_model_filename.pkl')

joblib.dump(scaler, 'your_scaler_filename.pkl')


# Building a Predictive System ( Xây dựng hệ thống dự đoán )

# In[40]:


input_data = (197.07600,206.89600,192.05500,0.00289)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#chuẩn hóa dữ liệu
df = scaler.transform(input_data_reshaped)

prediction = model.predict(df)
print(prediction)

if (prediction[0] == 0):
  print("Is a dangerous area.")
else:
  print("Not a dangerous area.")

