# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:51:52 2021

@author: brhng
"""

import pandas as pd 

veriler = pd.read_csv("Automobile.csv")

x = veriler.iloc[:,1:-1]
y = veriler.iloc[:,-1:]


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
ohe  = OneHotEncoder()
lb = LabelEncoder()


x.iloc[:,2:3] = lb.fit_transform(x.iloc[:,2:3])#fueltype

x.iloc[:,3:4] = lb.fit_transform(x.iloc[:,3:4])#aspiration

x.iloc[:,4:5] = lb.fit_transform(x.iloc[:,4:5])#doornumbet


carbody = x.iloc[:,5:6]

carbody = ohe.fit_transform(carbody).toarray()


drivewheel = x.iloc[:,6:7]

drivewheel = ohe.fit_transform(drivewheel).toarray()


x.iloc[:,7:8] = lb.fit_transform(x.iloc[:,7:8])#enginelocation

enginetype = x.iloc[:,13:14]

enginetype = ohe.fit_transform(enginetype).toarray()


x.iloc[:,14:15] = lb.fit_transform(x.iloc[:,14:15])#enginelocation

fuelsystem = x.iloc[:,16:17]

fuelsystem = ohe.fit_transform(fuelsystem).toarray()

carname = x.iloc[:,1:2]

carname = ohe.fit_transform(carname).toarray()


cars = ["carname"]*147
body = ["carbody"]*5
wheel = ["drivewheel"]*3
engine = ["enginetype"]*7
fuel = ["fuelsystem"]*8


carbody = pd.DataFrame(carbody,columns=body)
drivewheel = pd.DataFrame(drivewheel,columns=wheel)
enginetype = pd.DataFrame(enginetype,columns=engine)
fuelsystem = pd.DataFrame(fuelsystem,columns=fuel)
carname = pd.DataFrame(carname,columns=cars)

x_eksik = x.iloc[:,[2,11,12,15,17,18,19,20,21,23]]

s = pd.concat([carbody,drivewheel],axis=1)
s2 = pd.concat([enginetype,fuelsystem],axis=1)
s3 =pd.concat([s2,carname],axis=1)
s4 =pd.concat([s,fuelsystem],axis=1)

x_res =pd.concat([x_eksik,s4],axis=1) 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_res.iloc[:,1:10] = ss.fit_transform(x_res.iloc[:,1:10])
#y scale edilmez

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_res,y,test_size=0.33,random_state=100) 

from sklearn.linear_model import LinearRegression

lr =LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))

from sklearn.feature_selection import RFE
rf = RFE(lr,10)

rf = rf.fit(x_train,y_train)

x_train_rf = x_train[x_train.columns[rf.support_]]

import statsmodels.api as sm

x_ols = sm.add_constant(x_train)

model =sm.OLS(y_train,x_ols).fit()

print(model.summary())