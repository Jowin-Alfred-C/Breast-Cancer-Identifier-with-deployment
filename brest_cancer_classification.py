# -*- coding: utf-8 -*-
"""BREST CANCER CLASSIFICATION.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nycpy2HGBYRXrDdhfqkDBkjSnWwu4pZX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("/content/data.csv")
data

data.columns

data.shape

data.info()

data.isnull().sum()

dff=data.drop(columns='Unnamed: 32')

df=dff.drop(columns='id')

from sklearn import preprocessing 
label_encoder=preprocessing.LabelEncoder()

df['diagnosis']=label_encoder.fit_transform(df['diagnosis'])
df['diagnosis'].unique()

"""1 = m

0=b
"""

df

df['diagnosis'].value_counts()

df.groupby('diagnosis').mean()

x=df.drop(columns='diagnosis',axis=1)
y=df['diagnosis']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)

knn.score(X_test,Y_test)

from sklearn.metrics import accuracy_score
pred=knn.predict(X_train)
trainacc=accuracy_score(Y_train,pred)
trainacc

predd=knn.predict(X_test)
testacc=accuracy_score(Y_test,predd)
testacc

#logistic
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
lr.score(X_test,Y_test)

#random forest
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(X_train,Y_train)

classifier.score(X_test,Y_test)

from sklearn.svm import SVC # "Support vector classifier"  
svc = SVC(kernel='linear', random_state=0)  
svc.fit(X_train, Y_train)

svc.score(X_test,Y_test)

#save model
import joblib
joblib.dump(svc,"breast_cancer_classification.pkl")

#load model
load_model=joblib.load("breast_cancer_classification.pkl")
load_model.score(X_test,Y_test)

"""#-       C.JOWIN ALFRED"""