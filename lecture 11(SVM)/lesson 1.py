# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 03:20:44 2021

@author: brhng
"""

import pandas as pd 

df = pd.read_csv('bank-additional.csv', sep=';')
df = df.drop('duration',axis=1)
col_names = df.columns

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include='object').columns

df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})

df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

nominal = ['job','marital','education','contact','month','day_of_week']

df = pd.get_dummies(df,columns=nominal)
df['y']=df['y'].map({'yes': 1,'no': 0})

X = df.drop(['y'], axis=1)
y = df['y']

y= y.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

cols = X_train.columns


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# X_train üzerinde fit ve transform yap
X_train = scaler.fit_transform(X_train)
# X_test'i transform yap
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
y_pred= y_pred.reshape(-1,1)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(":",ac)

print("\nAccuracy :",svc.score(X_test,y_test))