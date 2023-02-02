# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 00:19:26 2021

@author: brhng
"""

import pandas as pd 

veriler = pd.read_csv("train.csv")
veriler2 = veriler.copy()

veriler2.dropna(axis=0,subset=["SalePrice"],inplace=True)
veriler2.drop(["LotFrontage","GarageYrBlt","MasVnrArea","Alley","PoolQC","MiscFeature"],axis=1,inplace=True)

x = veriler2.iloc[:,:-1]
y = veriler2.iloc[:,-1:]

numeric_cols = [cname for cname in x.columns if x[cname].dtype in ["int64","float64"]]

x = x[numeric_cols].copy()


from sklearn.model_selection import KFold

kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#n_splits =5  k değeri
#shuffle karıştırma
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(LinearRegression(),x,y,cv=kfold,scoring="neg_mean_squared_error")

mse = cvs.mean()

