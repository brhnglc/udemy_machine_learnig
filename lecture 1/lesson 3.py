        # -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:48:55 2021

@author: brhng
"""

import pandas as pd
import  matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("NCI60.csv",index_col=0)

x = df.iloc[:,:-1]

sc = StandardScaler()

x_scale = sc.fit_transform(x)

pca = PCA(n_components=2) #Principal Component Analysis feature seçmede

pca_result = pca.fit_transform(x_scale)

print(pca.explained_variance_) #eigenvalues
 
print(pca.explained_variance_ratio_*100)  #ilişkinin % kaçına karşılık geliyor seçilen ikili

principalDf = pd.DataFrame(data=pca_result,columns=["PC1","PC2"])

finalDf = pd.concat([principalDf,df["labs"]],axis =1)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("En etkili genler -hastalık")
targets =["COLON","MELANOMA","LEUKEMIA"]
ax.grid()



for target in targets:
    index = finalDf["labs"] == target
    ax.scatter(finalDf.loc[index,"PC1"],
               finalDf.loc[index,"PC2"], s= 50)


ax.legend(targets,loc="upper right")
