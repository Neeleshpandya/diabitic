# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:54:16 2022

@author: NH1305
"""

##Dataset Link: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#read csv file
df=pd.read_csv('diabetes.csv')

df

### Independent and Dependent features

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

### Train Test Split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

### Implement Random Forest classifier

classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

## Prediction
y_pred=classifier.predict(X_test)


### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

score

### Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

#predict

classifier.predict([[2,3,334,99,9996,66,8,66]])
