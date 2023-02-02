# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 01:15:23 2021

@author: brhng
"""
import numpy as np
import matplotlib.pyplot as plt
def veri_olustur():
    """
    Veriyi oluşturan fonksiyon.
    features: değişkenler, x ve y
    labels: sınıflar (A, B)
    """
    
    features = np.array(
        [[2.88, 3.05], [3.1, 2.45], [3.05, 2.8], [2.9, 2.7], [2.75, 3.4],
         [3.23, 2.9], [3.2, 3.75], [3.5, 2.9], [3.65, 3.6],[3.35, 3.3]])
    
    labels = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    
    return features, labels

features,labels = veri_olustur()

plt.scatter(features[5:,[0]],features[5:,[1]],c ="green") #A grubu
plt.scatter(features[:5,[0]],features[:5,[1]],c ="blue") #B grubu


#bulmak istenilen
plt.scatter([3.18],[3.15],c="r",marker="x")

