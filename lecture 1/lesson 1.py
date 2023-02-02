# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:53:20 2021

@author: brhng
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df =pd.read_csv("Wage.csv")

print(df.head(10))
#head defualt 5 parantez içine girilen değer kadar verinin satırını okur




#-----------------------------------"age","wage"
df1 = df[["age","wage"]]


x = df[["age"]] #bağımsız değişken-feature
y = df[["wage"]]  #bağımlı değişken-label


y_mean = df1.groupby("age").mean() #yaşlar index olup wage in ortalaması alınmış



plt.subplots(figsize=(8,8)) #grafigin boyutunu belirtiyor
plt.scatter(x,y)
plt.plot(y_mean.index,y_mean,c="red")
plt.title("MAAŞ-YAŞ")
plt.xlabel("Yaş")
plt.ylabel("Maaş")
plt.show()

#-----------------------------------"education","wage"


df1 = df[["education","wage"]]


x = df[["education"]] 
y = df[["wage"]] 

y_grouped = df1.groupby("education") 

y_grouped.boxplot(subplots=False,figsize=(8,8),rot=90)#rot altaki yazıların dönme derecesi

 



