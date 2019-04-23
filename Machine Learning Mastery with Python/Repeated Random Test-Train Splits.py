# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:51:30 2019

@author: C57946A
"""

import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)
array = dataframe.values
X= array[:,0:8]
Y = array[:,8]

n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv = kfold)
print('Acurracy: {}{} {}{}'.format(round(results.mean()*100,3),'%', round(results.std()*100,3),'%')) # std is the standard deviation