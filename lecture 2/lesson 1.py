# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:14:52 2021

@author: brhng
"""

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

df = pd.read_csv("Advertising.csv",index_col=0)

x = df.iloc[:,:-1]
y = df.iloc[:,-1:]




plt.scatter(x["newspaper"],y)
plt.title("News")
plt.show()

plt.scatter(x["radio"],y)
plt.title("radio")
plt.show()

plt.scatter(x["TV"],y)
plt.title("TV")


lr.fit(x.iloc[:,:1],y)
y_pred =lr.predict(x.iloc[:,:1])

plt.plot(x.iloc[:,:1],y_pred,c="red")