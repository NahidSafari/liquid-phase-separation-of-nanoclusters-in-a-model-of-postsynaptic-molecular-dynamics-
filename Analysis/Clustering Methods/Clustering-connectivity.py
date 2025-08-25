#!/usr/bin/env python
# coding: utf-8

#**************************************************************************************************
# ***  Clustering Using Connectivity matrix and Graph Theory  ***
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
import imageio

# ----------------------------------------
# Loading .xyz file containing the coordinates of particles over time 
# ----------------------------------------

concentration_value=0.0002
NAME='4_sites_fixed_'+str(concentration_value)+'.xyz'
with open (NAME, "r") as myfile:
    data = myfile.read().splitlines()
all_indexes = [] 
all_indexes2 = []
for i in range(0, len(data)) : 
    if len(data[i]) == 4 : 
        all_indexes.append(i)
        all_indexes2.append(i)
all_indexes2.append(len(data))
DS=np.diff(all_indexes2)


X0=[]
T=[]
T2=[]
HistD=[]
Cluster_size_N=[]
ims =[]
Vertex=[]
M=[]
S=[]
CV=[]
Rnge=[]

fig, ax = plt.subplots(figsize=(8, 6))

# ----------------------------------------
# A loop over all time snapshots
# ----------------------------------------


for l in range(1,2,1):
    # Extracting the coordinates of particles at time t 
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
    # ----------------------------------------    
    # Apply Clustering Method using Graph theory
    # ----------------------------------------
    
    # Create a histogram of the x-coordinates in Coordinate array
    hist, bin_edges = np.histogram(Coordinate[:, 0], bins=100)
    # Append the number of core atoms to list T
    T.append(len(Coordinate))

    # Initialize a connectivity matrix with zeros, where each element represents a connection between core particles
    Connectivity_Matrix = np.zeros((len(Cores), len(Cores)))
    # Initialize a list to store distances between core atoms
    Dis = []
    # Loop through each pair of core particles to calculate the Euclidean distance
    for r in range(len(Cores)):
        a = Coordinate[r]  # Get the coordinates of the r-th core atom
        for c in range(len(Cores)):
            b = Coordinate[c]  # Get the coordinates of the c-th core atom
            dst = distance.euclidean(a, b)  # Calculate Euclidean distance between atoms a and b
            # If the distance is greater than 0 and less than 6.05, consider them connected
            if dst > 0.0 and dst < 6.05:
                Dis.append(dst)  # Store the distance
                Connectivity_Matrix[r, c] += 1  # Update the connectivity matrix

    # Append the sum of connectivity values (degree) for each vertex (atom) to the Vertex list
    Vertex.append(sum(Connectivity_Matrix))
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    # Add edges to the graph where the connectivity matrix has non-zero values
    for ic in range(len(Cores)):
        for jc in range(len(Cores)):
            if Connectivity_Matrix[ic][jc] != 0:
                G.add_edge(ic, jc)  # Add a directed edge between core atom ic and jc

    # Get the weakly connected components of the graph (groups of connected atoms)
    aa = list(nx.weakly_connected_components(G))
    
    
    # Initialize a list to store the size of each cluster
    Cluster_size = []
    # Append the size of each weakly connected component to the Cluster_size list
    for i in range(len(aa)):
        Cluster_size.append(len(aa[i]))
        
    # Append the size of the largest cluster to Cluster_size_N
    Cluster_size_N.append(max(Cluster_size))
    # Calculate the relative size difference of clusters and append it to the list M
    M.append((max(Cluster_size) - min(Cluster_size)) / sum(Cluster_size))
    MeanA.append(np.mean(M))

