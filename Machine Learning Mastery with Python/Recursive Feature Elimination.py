# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:54:12 2019

@author: C57946A
"""
# Feature Extraction with RFE
"""Recursively removes attributes andn builds a model on those attributes that remain
Identify wich attributes contribute the most to predict the target attribute"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load data 

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)

array = dataframe.values

# Separate array into input and outcome components

X= array[:,0:8]
Y = array[:,8]

# Feature Extraction

model = LogisticRegression()
rfe = RFE(model,3) # Select the top 3 features
fit = rfe.fit(X,Y)

print('Num Features: {}'.format(fit.n_features_))
print('Selected Features: {}'.format(fit.support_))
print('Feature Ranking: {}'.format(fit.ranking_))

''' Output:
Num Features: 3
Selected Features: [ True False False False False  True  True False]
Feature Ranking: [1 2 3 5 6 1 1 4]
'''


