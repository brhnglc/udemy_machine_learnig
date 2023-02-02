# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:39:47 2021

@author: brhng
"""

import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("Income.csv",index_col=0)
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]

plt.figure(figsize=(10,6))
plt.scatter(x,y)




from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x,y)

y_pred = lr.predict(x) #bu verilere göre nasıl bir predict yapıyor  

plt.plot(x,y_pred)