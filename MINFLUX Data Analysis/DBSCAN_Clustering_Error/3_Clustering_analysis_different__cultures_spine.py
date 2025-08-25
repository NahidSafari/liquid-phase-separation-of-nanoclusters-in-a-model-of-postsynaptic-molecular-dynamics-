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


# This code performs clustering analysis on various datasets corresponding
#                                to different cultures and spine numbers.
# It uses the DBSCAN algorithm and evaluates the clustering results 
#        based on different epsilon values and minimum sample sizes.



# Initialization : Loading the id of cultures and spine numbers
cultures=[110,112,200,201,202,310,311,401,403,520,521,522]
spine_numbers = {
    110: np.arange(1, 10),
    112: np.arange(1, 12),
    200: np.arange(1, 13),
    201: np.arange(1, 12),
    202: np.arange(1, 8),
    310: np.arange(1, 24),
    311: np.arange(1, 21),
    401: np.arange(1, 14),
    403: np.arange(1, 26),
    520: np.arange(1, 23),
    521: np.arange(1, 23),
    522: np.arange(1, 31)
}

# Load Optimal values from Error using average eps 
eps_optimal=np.loadtxt('Optimal_eps_TTX.txt', delimiter=',')
mean_value = eps_optimal
STD = mean_value * 0.2 # Sets the standard deviation as 20% of the mean epsilon.

# Define Ranges for Epsilon and Min Samples 
eps_opt_range=np.linspace(mean_value - STD,mean_value + STD,10)
N_opt_range=np.arange(15,30,5) # Defines a range of minimum sample sizes for DBSCAN. 


# Main loop: Loops through each epsilon value and each minimum sample size
results = []
for i in range (len(eps_opt_range)):
    eps=eps_opt_range[i]
    for j in range(len(N_opt_range)):
        k=N_opt_range[j]
        eps_k_results = {'eps': eps, 'min_samples': k, 'data': []}
        # Nested Loop Through Cultures and Spines
        for culture in cultures:
            for spine_number in spine_numbers[culture]:
                file_name = f'{culture}cluster_{spine_number}.txt'
                points = np.loadtxt(file_name, delimiter=',')        
                min_samples = k  # minimum number of points in a cluster
                 # instantiate DBSCAN object with hyperparameters
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                # fit the model to the points
                dbscan.fit(points)
                # get the labels assigned by the model (i.e., the cluster assignments)
                labels = dbscan.labels_
                # print the labels for each point
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                Cluster_size = np.bincount(labels[labels >= 0])
                cluster_positions = {}
                Num_par_in_cluster=0
                Num_par_in_clusterL=[]
                for cluster_id in np.unique(labels):
                    if cluster_id != -1:
                        cluster_indices = np.where(labels == cluster_id)[0]
                        cluster_positions[cluster_id] = points[cluster_indices]
                        Num_par_in_cluster+=len(cluster_indices)
                        Num_par_in_clusterL.append(len(cluster_indices))

                noise_points = points[labels == -1]
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                num_noise_points = list(labels).count(-1)
#                 print(num_noise_points)
                M=np.mean(Num_par_in_clusterL)/(Num_par_in_cluster+num_noise_points)
                formatted_value2 = "{:.2f}".format(M)
                eps_k_results['data'].append({'file_name': file_name, 'num_clusters': n_clusters, 'mean': formatted_value2})
        results.append(eps_k_results)


        
# Save results to a file (e.g., CSV)
with open('cluster_results_NT.csv', 'w') as file:
    file.write('eps,min_samples,file_name,num_clusters,mean\n')
    for result in results:
        eps_val = result['eps']
        k_val = result['min_samples']
        for data in result['data']:
            file_name = data['file_name']
            num_clusters = data['num_clusters']
            mean_val = data['mean']
            file.write(f'{eps_val},{k_val},{file_name},{num_clusters},{mean_val}\n')        
        


# Load the CSV file into a DataFrame
df = pd.read_csv('cluster_results_NT.csv')
# Display the DataFrame
print(df)
# Load the CSV file into a DataFrame
df = pd.read_csv('cluster_results_NT.csv')
# Group the DataFrame by 'eps' and 'min_samples' and aggregate the 'mean' values
mean_values_dict = df.groupby(['eps', 'min_samples'])['mean'].apply(list).to_dict()
# Print the mean values for each combination of eps and N
for (eps, N), mean_values in mean_values_dict.items():
    print(f'Mean values for eps={eps}, N={N}: {mean_values}')


# Define the ranges for the histogram bins
ranges = [0.001, 0.45, 0.75, 1]
ii=0
# Iterate over each pair of eps and min_samples
for (eps, N), mean_values in mean_values_dict.items():
    ii+=1
    # Plot the histogram
    plt.hist(mean_values, bins=ranges, edgecolor='black', alpha=.9)
    # Calculate percentages
    total_samples = len(mean_values)
    percentages = [count / total_samples * 100 for count in np.histogram(mean_values, bins=ranges)[0]]
    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Values for eps={eps}, N={N}')
    # Annotate bars with percentages
    for i, percentage in enumerate(percentages):
        plt.text((ranges[i] + ranges[i + 1]) / 2, 0.9 * plt.ylim()[1], f'{percentage:.2f}%', ha='center')
    NAME=str(ii)+'.png'
    plt.savefig(NAME,dpi=1000)
    # Show the plot
    plt.show()


# Define the ranges for the histogram bins
ranges = [0.001, 0.45, 0.75, 1]

# Open a text file to write the percentages
with open("percentages_TTX.txt", "w") as f:
#     f.write("0.001-0.45\t0.45-0.75\t0.75-1\n")
    # Iterate over each pair of eps and min_samples
    for (eps, N), mean_values in mean_values_dict.items():
        # Calculate percentages
        f.write(f"{eps}\t{N}\t")
        total_samples = len(mean_values)
        percentages = [count / total_samples * 100 for count in np.histogram(mean_values, bins=ranges)[0]]

        # Write percentages to the text file
        f.write("\t".join([f"{percentage:.2f}" for percentage in percentages]))
        f.write("\n")


