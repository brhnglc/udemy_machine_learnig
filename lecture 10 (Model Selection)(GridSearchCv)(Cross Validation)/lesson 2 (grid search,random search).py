# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 02:30:27 2021

@author: brhng
"""

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

veriler = pd.read_csv("train.csv")

veriler.drop(["Name","Ticket","Cabin"],axis=1,inplace =True)

veriler["Age"].fillna(veriler["Age"].median(skipna=True),inplace=True) 

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

veriler["Sex"] = lb.fit_transform(veriler["Sex"])

veriler["Embarked"].fillna(veriler["Embarked"].value_counts().idxmax(),inplace=True)        
veriler["Embarked"] = lb.fit_transform(veriler["Embarked"])  

veriler.drop("PassengerId",axis=1,inplace=True)

x = veriler.iloc[:,1:]
y = veriler.iloc[:,:1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_pred =knn.predict(x_test)

from sklearn.metrics import f1_score,confusion_matrix

f1 = f1_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)

print("\naccuracy:",knn.score(x_test,y_test)*100,"\nf1:",f1*100,"\ncm\n",cm)



from sklearn.model_selection import GridSearchCV


gscv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={"n_neighbors":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]},cv=5,verbose=1,scoring="accuracy")

#cv = 5 k =5 oluyor kfold k kadar parça
#param_gird içindeki sayılar * cv  kadar modeli çalıştırıcak demek
#verbose bana sonuçları yaz

gscv.fit(x,y)
print(gscv.best_params_)
print(gscv.best_score_)
