# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:32:37 2021

@author: brhng
"""

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


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


y_mean =[y_test.mean()]*len(y_test)


plt.scatter(x_test,y_test)
plt.plot(x_test,y_mean,c="black")



lr2 = LinearRegression()
lr2.fit(x_train,y_train)
y_pred =lr2.predict(x_test)

plt.plot(x_test,lr2.predict(x_test),c="red")

y_v = y_test.values

sums_1 =0 
sums_2 =0 
for i in range(0,len(x_test)):
    sums_1 = sums_1+pow((y_pred[i]-y_v[i]),2)
    sums_2 = sums_2+pow((y_mean[i]-y_v[i]),2)   

    


print("R2 value:",1-(sums_1/len(x_test))/(sums_2/len(y_test)))
print("Real R2 value:",lr2.score(x_test,y_test))
#1-RSS/TSS

#RSS = Eyi-yî/n
