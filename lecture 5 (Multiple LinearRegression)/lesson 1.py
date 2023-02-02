# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:19:26 2021

@author: brhng
"""

import pandas as pd 
import numpy as np

veriler = pd.read_csv("Advertising.csv",index_col=0)

import matplotlib as plt
from sklearn.linear_model import LinearRegression

x = veriler.iloc[:,:-2]
y = veriler.iloc[:,-1:]

lr = LinearRegression()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=123)

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

import statsmodels.api as sm

x_train_ols = sm.add_constant(x_train.iloc[:,[0,1]]) # 1 eklenir

model = sm.OLS(y_train,x_train_ols).fit()

print(model.summary())

import seaborn as sns

sns.heatmap(veriler.corr(),annot =True)