# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:52:41 2021

@author: brhng
"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import seaborn as sns  plot gibi ama daha iyi çizim programı


veriler = pd.read_csv("Advertising.csv",index_col=0)

x = veriler.iloc[:,[0]] 
y = veriler.iloc[:,[-1]]

plt.scatter(x,y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print("r2:",lr.score(x_test,y_test))

from sklearn.metrics import r2_score ,mean_squared_error

r2 = r2_score(y_test,y_pred)
print("r2 %",r2*100)


mse = mean_squared_error(y_test,y_pred) # değerden ortalama sapmanız 2 ise 20 olan değere 22 18 filan demişsiniz gibi