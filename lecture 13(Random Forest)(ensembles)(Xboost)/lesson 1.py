# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 01:22:55 2021

@author: brhng
"""
import pandas as pd 
import numpy as np
import pandas as plt 


import warnings
warnings.filterwarnings("ignore")

veriler = pd.read_csv("train.csv")
veriler_first = veriler.copy()
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

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()

X_train = st.fit_transform(x_train)
X_test = st.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42,criterion="entropy",max_depth=4,max_features=None,n_estimators=10)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import roc_auc_score
rc = roc_auc_score(y_test,y_pred)
print("random forest:",rc)
"""
np.random.seed(42)

from sklearn.model_selection import GridSearchCV
param={"max_depth":[1,2,3,4,5,10,15],"criterion":["gini","entropy"],"max_features":["auto","sqrt","log2",None],"n_estimators":[10,100,200,300,400]}

gd  = GridSearchCV(estimator=rf,cv=5,param_grid=param,scoring="roc_auc")

gd.fit(X_train,y_train)

print("\nparams:",gd.best_params_)
print("\nscore:",gd.best_score_)
"""


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=400,learning_rate=1,random_state=42)

ada.fit(X_train,y_train)

y_pred2 = ada.predict(X_test)

rc2 = roc_auc_score(y_test,y_pred2)
print("ada boost:",rc2)


from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=400,max_depth=6,learning_rate=1,random_state=42)

xgb.fit(X_train,y_train)

y_pred3 = xgb.predict(X_test)

rc3 = roc_auc_score(y_test,y_pred3)
print("xgboost:",rc3)