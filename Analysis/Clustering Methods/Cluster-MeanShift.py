#!/usr/bin/env python
# coding: utf-8

#**************************************************************************************************
# ***  MeanShift Clustering Method ***
# **************************************************************************************************/
# Copyright Nahid Safari ***

import numpy as np
import scipy
from scipy.spatial import distance
from collections import Counter
import networkx as nx 
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from sklearn.cluster import MeanShift
import imageio

# Define a concentration value
concentration_value = 0.0001
# The file name  and file extension '.xyz'
NAME = '4_sites_fixed_' + str(concentration_value) + '.xyz'
# Open the file with the generated name in read mode
with open(NAME, "r") as myfile:
    # Read the file content and split it into lines
    data = myfile.read().splitlines()
# Initialize two empty lists to store the indexes
all_indexes = []
all_indexes2 = []
# Iterate through each line of the file
for i in range(0, len(data)):
    # If the length of the line is exactly 4 characters
    if len(data[i]) == 4:
        # Append the current index to both index lists
        all_indexes.append(i)
        all_indexes2.append(i)
# Append the length of the data list to all_indexes2 to capture the end of the file
all_indexes2.append(len(data))
# Calculate the differences between consecutive elements in all_indexes2
DS = np.diff(all_indexes2)

# Function to group consecutive numbers from a list into sublists
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

X0=[]
T=[]
T2=[]
HistD=[]
Cluster_size_N=[]
ims =[]
Vertex=[]
S=[]
CV=[]

# Loop through a range starting from 1 to len(all_indexes)-900
for l in range(1, len(all_indexes) - 900, 100):
    x = all_indexes[len(all_indexes) - l] + 2
    # Calculate the data size from DS and adjust it by subtracting 1
    Datasize = DS[len(DS) - l] - 1
    # Extract a subset of data between the calculated index and size
    data1 = data[x:x + Datasize]
    # Initialize an empty list to store indexes where the first character is 'C'
    Cores = []
    # Find all lines in data1 that start with 'C' (indicating core atoms)
    for i in range(len(data1)):
        if data1[i][0] == 'C':
            Cores.append(i)
    
    
    # Initialize a numpy array to store the coordinates of the core atoms
    Coordinate = np.zeros((len(Cores), 3))
    # Extract the coordinates for each core atom found in Cores
    for j in range(len(Cores)):
        tt = Cores[j]
        xx = []
        for i in range(len(data1[tt])):
            if data1[tt][i] == '\t':  # Assuming coordinates are tab-separated
                xx.append(float(data1[tt][i:i + 10]))
        Coordinate[j] = xx

        
    # Initialize MeanShift clustering model with a bandwidth (This parameter would affect the clustering considerably!)
    ms = MeanShift(bandwidth=22)
    # Fit the MeanShift model to the extracted core atom coordinates
    ms.fit(Coordinate)
    # Get the cluster labels and the centers of the clusters
    labels = ms.labels_
    centers = ms.cluster_centers_
    # Calculate the number of clusters based on unique labels
    num_clusters = len(np.unique(labels))
    
    # Initialize a list to store the sizes of the clusters
    Cluster_size = []
    # Calculate the size of each cluster by counting occurrences of each label
    for i in range(num_clusters):
        size = np.sum(labels == i)
        Cluster_size.append(size)
    

    # visualize clusters in 3D
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(num_clusters):
        # Get the points in this cluster
        cluster = Coordinate[labels == i]
        print(cluster)
        # Plot the points in this cluster
        ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], c=colors[i%len(colors)], marker='.')
        # Plot the cluster center
        ax.scatter(cluster_centers[i,0], cluster_centers[i,1], cluster_centers[i,2], c=colors[i%len(colors)], marker='x', s=100, linewidth=3)
    plt.title('Mean Shift Clustering')
    plt.show()

