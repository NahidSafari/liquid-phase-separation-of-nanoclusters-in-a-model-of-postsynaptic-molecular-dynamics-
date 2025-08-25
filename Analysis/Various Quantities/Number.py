#!/usr/bin/env python
# coding: utf-8

#**************************************************************************************************
# ***  Variation of the number of particles over time  ***
# **************************************************************************************************/
# Copyright Nahid Safari ***

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Loading .xyz file containing the coordinates of particles over time 
# ----------------------------------------

concentration_value=0.0002
NAME='4_sites_DC_'+str(concentration_value)+'.xyz'

with open (NAME, "r") as myfile:
    data = myfile.read().splitlines()
all_indexes = [] 
all_indexes2 = []

# ----------------------------------------
# File Reading and Index Extraction 
# ----------------------------------------

for i in range(0, len(data)) : 
    if len(data[i]) == 4 : 
        all_indexes.append(i)
        all_indexes2.append(i)
all_indexes2.append(len(data))

# ----------------------------------------
# Coordinate Array for all time snapshots: DS
# ----------------------------------------
DS=np.diff(all_indexes2)


# ----------------------------------------
# A loop over all time snapshots
# ----------------------------------------

for l in range(1,len(all_indexes),1):
    # Extracting the coordinates of particles at time t 
    # l=1 is equivalent to the last timestep!
    xx=all_indexes[len(all_indexes)-l]+2
    T1.append(xx)
    Datasize=DS[len(DS)-l]-1
    data1=data[xx:xx+Datasize]
    Cores=[]
    for ii in range(len(data1)):
        if data1[ii][0]=='C':
            Cores.append(ii)

    Coordinate=np.zeros((len(Cores),3))
    for j in range(len(Cores)):
        tt=Cores[j]
        xx=[]
        for i in range(len(data1[tt])):
            if data1[tt][i]=='\t':
                xx.append(float(data1[tt][i:i+10]))  
        Coordinate[j]=xx
    # len(Coordinates) shows the number of particles in the system!
    T.append(len(Coordinate))
    T=T[::-1]    
    
    # Visualize the variations in the number of particles
    plt.plot(T)
    plt.ylabel('Number of particles')
    plt.xlabel('t')
    plt.title(r'Change in the number of particles while $\epsilon$ increases')
    # plt.savefig("Number.png", format='png', figsize=(8, 6), dpi=1000)
    plt.show()



