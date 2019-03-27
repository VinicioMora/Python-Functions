# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:12 2019

@author: C57946A
"""

import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load Data

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

# Feature extraction
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X,Y)

# Sumarize Scores
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)

#Sumerize selected features
#print(features[0:5,:])