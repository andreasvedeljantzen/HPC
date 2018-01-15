# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 07:38:17 2018

@author: andre
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%

f = open('data.txt', 'r')
content = f.read()
print(content)
f.close()

#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Generate data:
x, y, z = content

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.scatter(x, y, c=z)
plt.colorbar()
plt.show()

#%%

import csv
with open("data.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)
firstColumn = [row[0] for row in d]


