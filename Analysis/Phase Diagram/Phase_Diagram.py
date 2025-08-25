#!/usr/bin/env python
# coding: utf-8
#**************************************************************************************************
# ***  Phase Diagram Plot  ***
# **************************************************************************************************/
# Copyright Nahid Safari ***

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animatio
from matplotlib.colors import ListedColormap
import numpy as np


# Create a list of values for eps and concentration
eps=[13.8,13.9,14.1,14.2,14.25,14.3,14.4,14.6,14.7,14.8,14.9,15.0,15.4,16.4,16.7,17.0]
Conc=[0.0001,0.0002,0.00026,0.00028,0.0003,0.0004,0.0005,0.0007,0.0008,0.00087,0.00093,0.001,0.002,0.005,0.007,0.008,0.01,0.015]

# Make an array out of eps and c
eps_array=[]
for j in range(len(eps)):
    for i in range (len(Conc)):
        eps_array.append(eps[j])

file=open('k.txt','w')
for i in range(len(eps_array)):
    txt=str(eps_array[i])+'\n'
    file.write(txt)
file.close()

# 3d Values for eps, c and phase id
Y1=np.loadtxt('k.txt')
X1=Conc
X , Y = np.meshgrid(X1,Y1)
N=[]
R=[]
for i in range(len(eps)):
    NAME='Phase_diagram_Value'+str(eps[i])+'.txt'
    z1=np.loadtxt(NAME)
    for j in range(len(Conc)):
        R.append(z1)
Z=np.array(R)

# --------------------------------------
# Visualization using two different Methods
# --------------------------------------

fig, ax1 = plt.subplots(figsize=(10,8))
plt.xlim(0.0001,0.001)
plt.csfont={'fontname':'Time New Roma'}

# Method 1
c = ax1.pcolormesh(X, Y, Z, shading='auto')



# Method 2 
# Define specific levels for contouring
levels = [-1, 0.0,0.5,0.82,1.0,1.99, 2,3.01]  # Specify the levels you want
custom_cmap = ListedColormap(['green','black','black','yellow','black', 'blue', 'red' ])
# Plot contour with specified levels
img = ax1.contourf(X, Y, Z, levels=levels,cmap=custom_cmap, extend='both', shading ='flat')






# Setting the features of the plot (tick parameters, labels, annotations, and limits)
ax1.tick_params(axis='x',which='minor',top='off')
ax1.tick_params(axis='y',which='major',right='off')
ax1.tick_params(axis='y',which='minor',right='off')
plt.ylabel(r'$\epsilon\ (Interaction\ strength)$',fontsize=30)
plt.xlabel(r'$C_{env}\ (Reservoir\ concentration)$',fontsize=30)
plt.xlim(0.0001,0.001)
# plt.ylim(13.5,17.2)
plt.annotate(r'$\mathit{MC}\ $', xy=(80,370), xycoords='axes points',
            size=25, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))

plt.annotate(r'$\mathit{SC}\ $', xy=(80,120), xycoords='axes points',
            size=25, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))

plt.annotate(r'$\mathit{0C}\ $', xy=(80,40), xycoords='axes points',
            size=25, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
plt.annotate(r'$\mathit{LC}\ $', xy=(350,310), xycoords='axes points',
            size=25, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
plt.annotate(r'$\mathit{MC}\ $', xy=(300,90), xycoords='axes points',
            size=25, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
ax1.set_xticks([0.0002,0.0006,0.001])
ax1.set_yticks([14.0,15.0,16.0,17.0 ])    
ax1.set_xticklabels([0.0002,0.0006,0.001], fontsize=19)
ax1.set_yticklabels([14.0,15.0,16.0,17.0 ], fontsize=19)    
# plt.savefig("PhaseDiagram.png", format='png', figsize=(8, 6), dpi=300)
plt.savefig("PhaseDiagram.svg", format='svg', figsize=(8, 6), dpi=300)

plt.show()

