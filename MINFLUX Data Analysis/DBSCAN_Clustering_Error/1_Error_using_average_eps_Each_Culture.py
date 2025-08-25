#!/usr/bin/env python
# coding: utf-8


#**************************************************************************************************
# ***     DBSCAN clustering Method, Error Estimation    ***
# **************************************************************************************************/
#  Copyright Nahid Safari ***


import scipy.io
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import  estimate_bandwidth
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def gaussian(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)


# Initialization
Number_Cluster=np.arange(1,31,1) # represents different clusters in each culture
Eps=[]
# The loop iterates over different clusters 
for i in range(len(Number_Cluster)):
    # Load the coordinates of PSD95 proteins in each spine
    C_spine= np.loadtxt('522cluster_'+str(Number_Cluster[i])+'.txt', delimiter=',')
    LM=C_spine
    # Nearest Neighbors Calculation// Elbow Detection
    k=2
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(LM)
    distances, indices = nbrs.kneighbors(LM)
    distances = distances[:, 1:]
    distances = np.sort(distances, axis=None)
    deriv = np.diff(distances, 1)
    window_size = 1
    deriv_smooth = np.convolve(deriv, np.ones(window_size)/window_size, mode='valid')
    deriv2 = np.diff(deriv_smooth, 1)
    elbow_idx = np.argmax(deriv2) + window_size//2
    eps0 = distances[elbow_idx]
    eps0 = eps0/2 # maximum distance between points in a cluster 
    # So for each spine, one optimal eps is stored considering k=2 for NN and elbow technique
    Eps.append(eps0)

# visualize the distribution of the epsilon values obtained from the clustering analysis. 
# By plotting a histogram and overlaying a Gaussian distribution, 
#          it provides insights into the central tendency and variability of the epsilon values.
# The use of mean and standard deviation helps in understanding 
#          how the epsilon values are distributed and can be useful for further analysis or comparisons.


# Calculate mean and standard deviation
mean_value = np.mean(Eps)
STD = mean_value * 0.33

# Plot histogram
plt.hist(Eps, bins=20, density=True, alpha=0.5, label='Histogram')

# Plot Gaussian distribution
x = np.linspace(min(distances), max(distances), 1000)
y = gaussian(x, mean_value, STD)
plt.plot(x, y, color='red', label='Gaussian Distribution')

# Add labels and legend
plt.title('Histogram and Gaussian Distribution: TTX')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.axvline(mean_value + STD, color='green', linestyle='dashed', linewidth=2, label='Mean + STD')
plt.axvline(mean_value - STD, color='green', linestyle='dashed', linewidth=2, label='Mean - STD')

plt.text(mean_value, max(y), f'Mean: {mean_value:.10f}', color='blue', fontsize=12, ha='center')

# Show plot
plt.grid(True)
plt.savefig('Histogram and Gaussian Distribution-TTX522.png',dpi=1000)
plt.show()



file=open('Opt522.txt','w')
for i in range(len(A)):
    txt=str(A[i])+'\n'
    file.write(txt)
file.close() 


