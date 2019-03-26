# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:08:20 2019

@author: C57946A
"""
import pandas as pd

filename = 'PID.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename, names = names)
class_counts = data.groupby('class').size()
print(class_counts)

