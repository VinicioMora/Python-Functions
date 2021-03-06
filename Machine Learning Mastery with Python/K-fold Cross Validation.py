# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:25:23 2019

@author: C57946A
"""

import pandas as pd
from sklearn.model_selection import KFold
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

num_folds = 10
seed = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression()

results = cross_val_score(model, X,Y, cv = kfold)
print('Acurracy: {}{} {}{}'.format(round(results.mean()*100,3),'%', round(results.std()*100,3),'%')) # std is the standard deviation