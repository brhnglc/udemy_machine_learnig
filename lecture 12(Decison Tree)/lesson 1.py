# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 03:20:44 2021

@author: brhng
"""

import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

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


df = pd.get_dummies(df,columns=nominal,drop_first=True)
df['y']=df['y'].map({'yes': 1,'no': 0})

X = df.drop(['y'], axis=1)
y = df['y']

y= y.values.reshape(-1,1)

X.drop(["emp.var.rate","euribor3m","cons.price.idx"],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

cols = X_train.columns

kor = X_train.corr()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# X_train üzerinde fit ve transform yap
X_train = scaler.fit_transform(X_train)
# X_test'i transform yap
X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


y_test = pd.DataFrame(y_test)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="gini")
 
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

from sklearn.metrics import roc_auc_score
rc = roc_auc_score(y_test,y_pred)


from sklearn.model_selection import GridSearchCV

param={"criterion":["gini","entrpoy"],"max_depth":[1,2,3,4,5,10,15]}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid=param,cv=5,scoring="roc_auc")

gs.fit(X_test,y_test)

print("\nparams:",gs.best_params_)
print("\nbest score:",gs.best_score_)




