#!/usr/bin/env python
# coding: utf-8
#**************************************************************************************************
# ***  DBSCAN Clustering Method using elbow method to determine eps  ***
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
    # ----------------------------------------
    # Applying DBSCAN clustering Method 
    # ----------------------------------------  
    
    # ----------------------------------------
    # Elbow method to determine the best eps
    # ---------------------------------------- 
    
    
    # Set the number of neighbors to 2 (used for finding nearest neighbors)
    k=2 
    # Fit a NearestNeighbors model to the 'Coordinate' data
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Coordinate)
    # Compute distances and indices of the k+1 nearest neighbors for each point
    distances, indices = nbrs.kneighbors(Coordinate)
    distances = distances[:, 1:]
    distances = np.sort(distances, axis=None)
    # find the first derivative of the distribution and smooth it using a rolling average
    deriv = np.diff(distances, 1)
    # Define the window size for the rolling average
    window_size = 10
    # Compute the second derivative of the smoothed first derivative
    deriv_smooth = np.convolve(deriv, np.ones(window_size)/window_size, mode='valid')
    # Find the index of the "elbow" (maximum of second derivative)
    deriv2 = np.diff(deriv_smooth, 1)
    elbow_idx = np.argmax(deriv2) + window_size//2
    # Determine the epsilon value at the elbow point (distance threshold)
    eps = distances[elbow_idx]
    eps = eps/2  # maximum distance between points in a cluster 
    
    
    # minimum number of points in a cluster for DBSCAN clustering
    min_samples = 20  
     # instantiate DBSCAN object with hyperparameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # fit the model to the points
    dbscan.fit(Coordinate)
    # get the labels assigned by the model (i.e., the cluster assignments)
    labels = dbscan.labels_
    # print the labels for each point
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    Cluster_size = np.bincount(labels[labels >= 0])
    
    Position=Coordinate
    cluster_positions = {}
    for cluster_id in np.unique(labels):
        if cluster_id != -1:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_positions[cluster_id] = Position[cluster_indices]
            
    # Visualize clusters and noise points in 3D color-coded plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through each cluster label and plot particles with that label
    # for cluster_id in np.unique(labels):
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_positions[cluster_id] = Position[cluster_indices]
            ax.scatter(cluster_positions[cluster_id][:, 0], cluster_positions[cluster_id][:, 1], cluster_positions[cluster_id][:, 2], label='Noise', c='gray', alpha=0.5)
        else: 
            print(len(cluster_positions[cluster_id]))
            ax.scatter(cluster_positions[cluster_id][:, 0], cluster_positions[cluster_id][:, 1], cluster_positions[cluster_id][:, 2], label=f'Cluster Size= {len(cluster_positions[cluster_id])}')
    formatted_outcome = "{:.2f}".format(np.mean(Cluster_size)/len(Coordinate))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('M='+str(formatted_outcome)+', SMC')
    ax.legend()
    plt.savefig('FigureName_3d.png',dpi=300)
    plt.show()
    
    # ----------------------------------------
    # Different Visualization: 2d
    # ---------------------------------------- 
    
    # Create the figure and axis objects with specified background colors
    fig, ax = plt.subplots(figsize=(5, 2))
    # fig.patch.set_facecolor('lightgray')  # Set the figure background color
    # Set the background color for the figure and axes
    ax.set_facecolor('lightblue')         # Set the axes background color

    # Increase the size of the points by setting the 's' parameter
    scatter = ax.scatter(Coordinate[:, 0], Coordinate[:, 1], c=labels, cmap='GnBu', marker='o', edgecolors='black', s=70)

    # Set the title
    ax.set_title('A Dilute and Dense Phase Coexistence State')

    # Remove ticks and labels
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Adjust the layout to make the box smaller
    plt.tight_layout(pad=0)

    # Save the figure with a tight bounding box
    plt.savefig("FigureName_2d.png", format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


