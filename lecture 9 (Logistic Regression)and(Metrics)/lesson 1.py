# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:51:09 2021

@author: brhng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



veriler = pd.read_csv("train.csv")

veriler2 = veriler.drop(["Name","Ticket","Cabin"],axis=1,inplace=False)

"""
mean = veriler2.describe()
mean_age = 29.6991
for i in range(0,len(veriler2["Age"])):
    if (np.isnan(veriler2["Age"][i])):
        veriler2["Age"][i] =mean_age
"""
veriler2["Age"].fillna(veriler2["Age"].median(skipna=True),inplace=True)        
        
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

veriler2["Sex"] = lb.fit_transform(veriler2["Sex"])       

"""
veriler2["Embarked"] = lb.fit_transform(veriler2["Embarked"]) 
freakans = veriler2.groupby("Embarked").size()
for i in range(0,len(veriler2["Embarked"])):
    if(veriler2["Embarked"][i] == 3):
        veriler2["Embarked"][i] =2
"""

veriler2["Embarked"].fillna(veriler2["Embarked"].value_counts().idxmax(),inplace=True)        
veriler2["Embarked"] = lb.fit_transform(veriler2["Embarked"])         

veriler2.drop("PassengerId",axis=1,inplace=True)    

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
a = ohe.fit_transform(veriler2.iloc[:,1:2]).toarray()
a = pd.DataFrame(a[:,:-1],columns=["Pclass1","Pclass2"])
veriler2.insert(2,"Pclass1",a["Pclass1"])
veriler2.insert(3,"Pclass2",a["Pclass2"])
veriler2.drop("Pclass",axis=1,inplace=True)


x = veriler2.iloc[:,1:]
y = veriler2.iloc[:,:1]

#import seaborn  as sb
#sb.heatmap(x.corr(),annot =True)
        
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler() 

x_train.iloc[:,3:-1] = ss.fit_transform(x_train.iloc[:,3:-1])   
x_test.iloc[:,3:-1] = ss.transform(x_test.iloc[:,3:-1])        


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="liblinear",random_state=0)

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("\nConfuison matrix: \n",cm)


from sklearn.metrics import roc_auc_score
rxx = roc_auc_score(y_test, y_pred)
print("\nROC-AUC Score(AUC):",rxx)


print("\nAccuracy:",lr.score(x_test,y_test))


dead = y_test.groupby("Survived").size()[0]
survived = y_test.groupby("Survived").size()[1]

print("\nNull Accuracy:",dead/(dead+survived),survived/(dead+survived))#,"büyük olan accuracy"

from sklearn.metrics import roc_curve
rc = roc_curve(y_test,y_pred)

plt.plot(rc[0],rc[1],label="ROC Curve")
plt.plot([0,1],[0,1],"k--",label="Random Guess")
plt.legend(loc="lower right")
plt.xlabel("TPRate")
plt.ylabel("FPRate")
plt.title("ROC Curve")

from sklearn.metrics import f1_score
f1 = f1_score(y_test,y_pred)
print("\nF1 Score",f1)


from sklearn.metrics import log_loss
lg = log_loss(y_test, y_pred)
print("\nLog-Loss:",lg)#0 a ne kadar yakınsa o kadar iyi