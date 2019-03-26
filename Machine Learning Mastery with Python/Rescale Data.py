# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:52:45 2019

@author: C57946A
"""

import pandas as pd
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array =  dataframe.values

# Separate array into input and output components

X = array[:,0:8]
Y = array[:,8]

scaler = MinMaxScaler(feature_range = (0,1))
rescaledX = scaler.fit_transform(X)

#Sumarize transformed data

set_printoptions(precision = 3)
print(rescaledX[0:5,:])

