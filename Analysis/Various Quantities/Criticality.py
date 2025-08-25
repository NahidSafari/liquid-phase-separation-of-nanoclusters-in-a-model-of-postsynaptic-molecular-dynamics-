#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8
#**************************************************************************************************
# ***  Criticality  ***
# **************************************************************************************************/
# Copyright Nahid Safari ***

import scipy.io
import numpy as np
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import  estimate_bandwidth


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



X0=[]; T=[]; T2=[]; HistD=[]; Cluster_size_N=[]; ims =[]; Vertex=[]; M=[]; S=[]; CV=[]; Rnge=[]

# ----------------------------------------
# A loop over all time snapshots
# ----------------------------------------

for l in range(1,2,1): # l: time snapshot interator!
# Extracting the coordinates of particles at time t 
# l=1 is equivalent to the last timestep! 
    x=all_indexes[len(all_indexes)-l]+2
    Datasize=DS[len(DS)-l]-1
    data1=data[x:x+Datasize]
    Cores=[]
    for i in range(len(data1)):
        if data1[i][0]=='C':
            Cores.append(i)
    Coordinate=np.zeros((len(Cores),3))
    for j in range(len(Cores)):
        tt=Cores[j]
        xx=[]
        for i in range(len(data1[tt])):
            if data1[tt][i]=='\t':
                xx.append(float(data1[tt][i:i+10]))  
        Coordinate[j]=xx
    # Apply One of the clustering Methods
    # ....
    # ....
    # ....
    # ....
    # ....
    # So when we have the Cluster_size, we can simply explore criticality
    M.append((np.mean(Cluster_size)))
    MeanA.append(np.mean(M))
    Cluster_size.sort()
    q=Counter(Cluster_size)
    pp=list(q.values())
    nt=list(q.keys())
    a=max(nt)
    nt.sort()
    p=[]
    for i in nt:
        ee=q[float(i)]
        p.append(ee)
    p=np.array(p)
    s=float(sum(p))
    p=p/s
    # If the logarithm scale plot is a line so it shows a powerlaw behavior
    plt.plot(nt,p,'--bo',ms=8,linewidth=2.5)
    plt.yscale('log')
    plt.xscale('log')
    Name='eps='+str(ii)
    plt.title('eps=17.0, c=0.0002')
    Name=str(ii)+'.png'
    plt.savefig(Name)
    plt.show()

