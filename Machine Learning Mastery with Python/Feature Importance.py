# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:28:40 2019

@author: C57946A
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
# Load data 

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

# Feature Extraction
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)



