# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:53:37 2021

@author: brhng
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df =pd.read_csv("Smarket.csv")

print(df.describe()) #basit istatistik 

df4 = df[["Today","Direction"]]

df4[["Today"]]=  df4[["Today"]].abs()

y_grouped = df4.groupby("Direction")

y_grouped.boxplot(subplots=False,figsize=(8,8),rot=90)

print(y_grouped.count())
#4:40
