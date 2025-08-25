#!/usr/bin/env python
# coding: utf-8
#**************************************************************************************************
# ***  Jaccard Index to explore the stability of Clusters over time  ***
# **************************************************************************************************/
# Copyright Nahid Safari ***


import numpy as np
import scipy
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx 
from random import randint
import matplotlib.animation as animation
import plotly.express as px
import scipy.stats as st
from sklearn.neighbors import KernelDensity
import cv2
from sklearn.metrics import jaccard_score
from sklearn.model_selection import GridSearchCV

def group_consecutive_numbers(numbers):
    sorted_numbers = sorted(numbers)
    groups = []
    current_group = [sorted_numbers[0]]
    for number in sorted_numbers[1:]:
        if number == current_group[-1] + 1:
            current_group.append(number)
        else:
            groups.append(current_group)
            current_group = [number]
    groups.append(current_group)
    return groups

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
bandwidth=2.0
Count=0
for l in range(1,len(all_indexes)-1,1): # l: time snapshot interator!
# Extracting the coordinates of particles at time t 
# l=1 is equivalent to the last timestep! 
    Count+=1
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
    # Filter non-zero coordinates
    Core_Coordinate=[]
    for c in range(len(Coordinate)):
        if Coordinate[c,0]!=0.0 and Coordinate[c,1]!=0.0 and Coordinate[c,2]!=0.0:
            Core_Coordinate.append(Coordinate[c])
    # Determine the cores in a specific range along the z-axis ...> Just to make it 2D! 
    First_snapshot=[]
    for ff in range(len(Cores)):
        ff=int(ff)
        if Core_Coordinate[ff][2]>=-80.0 and Core_Coordinate[ff][2]<=80.0:
            First_snapshot.append(ff)
    # Extract x and y coordinates:
    X=[]; Y=[]
    for g in range(len(Coordinate)):
        X.append(Core_Coordinate[g][0])
        Y.append(Core_Coordinate[g][1])
    points = np.zeros([len(X),2])
    points[:,0]=X; points[:,1]=Y
    # Kernel Density Estimation (KDE)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    # Evaluate KDE on a grid of points:
    xmin, ymin = points.min(axis=0) - 0.5
    xmax, ymax = points.max(axis=0) + 0.5
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    zz = np.exp(kde.score_samples(np.column_stack([xx.ravel(), yy.ravel()])))
    zz = zz.reshape(xx.shape)
    # Plot the KDE as a contour plot
    plt.contourf(xx, yy, zz, cmap=plt.cm.inferno)
    plt.savefig(Name)
    plt.show()

# --------------------------
# Compute the Jaccard Index between consecutive images saved in previous Loop!
# --------------------------

Similarity=[]
# Loop through a range of image indices
for l in range(1,len(all_indexes)-1,1): # l: time snapshot interator!
    NAME1=str(l)+'.png'
    img1 = cv2.imread(NAME1)
    NAME2=str(l+1)+'.png'
    img2 = cv2.imread(NAME2)
    img_true=np.array(img1).ravel()
    img_pred=np.array(img2).ravel()
    # Compute the Jaccard similarity (IoU) between the two images
    iou = jaccard_score(img_true, img_pred,average='micro')
    Similarity.append(iou)
    
    
Similarity=np.array(Similarity)
plt.plot(Similarity,'o')
plt.ylim([0,1])

