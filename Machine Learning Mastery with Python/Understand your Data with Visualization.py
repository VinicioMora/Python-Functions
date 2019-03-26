# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:02:08 2019

@author: C57946A
"""

import pandas as pd
import matplotlib as mt

filename = 'PID.csv' # CSV FILE
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(filename, names = names)

# Histograms

data.hist()
mt.pyplot.show()

# Density Plots

data.plot(kind='density', subplots = True, layout = (3,3), sharex = False)
mt.pyplot.show()

# Box and Whisker Plots
data.plot(kind='box', subplots = True, layout = (3,3), sharex = False, sharey = False)
mt.pyplot.show()

### Multivariate Plots
# * Corelation Matrix Plot
#* Scatter Plot Matrix

# Correlation Matrix Plot
#   Correlation gives an indication of how related the changes are between two variables
#   If two variables change in the same direction they are positivele correlated.
#    If they change in opposite directions together(one up, one down), they are negatively correlated
import numpy as np

correlations = data.corr()
fig = mt.pyplot.figure() # <- Plot correlation matrix
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)

ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xticklabels(names)
ax.set_yticklabels(names)
mt.pyplot.show()
# The matrix is symetrical, the botom left of the matrix is the same as the top right

# Scatter Plot Matrix
#   Shows the relationship between two variables as dots in two dimensions, one axis for each attribute 
from pandas.tools.plotting import scatter_matrix
scatter_matrix(data)
mt.pyplot.show()


