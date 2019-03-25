# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:55:57 2019

@author: C57946A
"""

import pandas as pd
from pandas import set_option

filename = 'pima indians diabetes.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename, names = names)
data.head(20)

shape = data.shape
print(shape)

types = data.dtypes
print(types)

set_option('display.width',100)
set_option('precision',3)
description = data.describe()
print(description)


claim = "Pluto is a planet!"
print(claim.startswith('Pluto'))
print(claim.endswith('dwarf planet'))



datestr = '1956-01-31'
def ChangeDateFormat(date):

    year, month, day = date.split('-')    
    return '/'.join([month, day, year])

ChangeDateFormat(datestr)