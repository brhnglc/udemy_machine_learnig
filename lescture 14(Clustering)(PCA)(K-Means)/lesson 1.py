# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 03:20:00 2021

@author: brhng
"""

from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
x=iris.data


pca = PCA(n_components=2)  



x_pca = pca.fit_transform(x)

"""
x_centered = x-x.mean(axis=0)
"""



from sklearn.cluster import KMeans

wcss= []
for i in range(1,11):
    km = KMeans(n_clusters=i,init="k-means++")
    km.fit(x)
    wcss.append(km.inertia_)
    

import seaborn as sns

sns.lineplot(range(1,11),wcss,marker="o")    



km = KMeans(n_clusters=4,init="k-means++")
km.fit(x)

y_kmean = km.predict(x)