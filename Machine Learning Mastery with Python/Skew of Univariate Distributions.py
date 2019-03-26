# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:50:33 2019

@author: C57946A
"""

import pandas as pd
from pandas import set_option

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename, names = names)

skew = data.skew()
print(skew)

#The skew result show a positive (right) or negative (left) skew. values closer to zero show less skew.

