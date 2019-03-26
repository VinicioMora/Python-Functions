# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:21:18 2019

@author: C57946A
"""

# Binarization
from sklearn.preprocessing import Binarizer
import pandas as pd
from numpy import set_printoptions

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

binarizer = Binarizer(threshold = 0.0).fit(X)
binaryX = binarizer.transform(X)

# Sumarize transformed data
set_printoptions(precision = 3)
print(binaryX[0:5,:])

print(binaryX[0,0])