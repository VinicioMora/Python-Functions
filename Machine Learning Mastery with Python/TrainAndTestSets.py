# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:53:38 2019

@author: C57946A
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pd.read_csv(filename, names = names)
array = dataframe.values
X= array[:,0:8]
Y = array[:,8]

test_size = 0.33
seed = 8
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=test_size, random_state = seed)

model = LogisticRegression()
model.fit(X_train,Y_train)
result = model.score(X_test, Y_test)
print('Accuracy: {}{}'.format(round(result*100,3),'%'))