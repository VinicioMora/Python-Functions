# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:15:43 2019

@author: C57946A
"""

# Standariza data (0 mean, 1 stdev)

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# Sumarize transformed data

np.set_printoptions(precision = 3)
print(rescaledX[0:5,:])

