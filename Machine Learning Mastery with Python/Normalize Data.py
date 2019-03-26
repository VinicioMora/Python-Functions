# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:12:48 2019

@author: C57946A
"""

from sklearn.preprocessing import Normalizer
import pandas as pd
from numpy import set_printoptions

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

# Sumarize transformed data
set_printoptions(precision = 3)
print(normalizedX[0:5,:])