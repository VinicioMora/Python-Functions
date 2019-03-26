# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:33:04 2019

@author: C57946A
"""

import pandas as pd
from pandas import set_option

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename, names = names)

set_option('display.width',100)
set_option('precision',3)

correlations = data.corr(method = 'pearson')
print(correlations)

# Pearlson's Correlation Coefficient
# A correlation of -1 or 1 shows full negative or positive correlation
# A correlation of 0 shows no correlation at all