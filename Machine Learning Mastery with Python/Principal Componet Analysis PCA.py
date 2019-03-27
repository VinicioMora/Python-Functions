# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:21:31 2019

@author: C57946A
"""

import pandas as pd
from sklearn.decomposition import PCA

# Load data 

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

pca = PCA(n_components =3)  # Select the 3 principal components
fit =pca.fit(X)
 
# Sumarize components

print ('Explained Variance: {}'.format(fit.explained_variance_ratio_))
print(fit.components_c)