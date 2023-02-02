# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:47:34 2021

@author: brhng
"""

import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("iris.csv")

x = veriler.iloc[:,1:-1]
y = veriler.iloc[:,-1:]

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y) 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=4)
     
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) #
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print(knn.score(x_test,y_test)*100)

"""
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)) 
"""

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)