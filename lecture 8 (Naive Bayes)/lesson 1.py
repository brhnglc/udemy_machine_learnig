# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:51:29 2021

@author: brhng
"""

import pandas as pd

veriler = pd.read_csv("wine.data",header=None)

x = veriler.iloc[:,1:]
y = veriler.iloc[:,:1]

#x.columns = ["kolonlara isin verilir burada"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

from sklearn.naive_bayes import GaussianNB #gaussian naive bayes,bernoulli multinomial

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)
print(gnb.score(X_test, y_test))  # bu yüksek çıkarsa öbürküne göre underfitting
print(gnb.score(X_train, y_train)) # bu yüksek çıkarsa öbürküne göre overfitting

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
