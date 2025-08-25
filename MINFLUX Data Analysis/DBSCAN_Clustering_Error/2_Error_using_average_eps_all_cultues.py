#!/usr/bin/env python
# coding: utf-8
#**************************************************************************************************
# ***     DBSCAN clustering Method, Error Estimation    ***
# **************************************************************************************************/
#  Copyright Nahid Safari ***

# Load the optimal epsilon values from different text files for various cultures

AA=np.loadtxt('Opt110.txt')
BB=np.loadtxt('Opt112.txt')
CC=np.loadtxt('Opt200.txt')
DD=np.loadtxt('Opt201.txt')
EE=np.loadtxt('Opt202.txt')
FF=np.loadtxt('Opt310.txt')
GG=np.loadtxt('Opt311.txt')
KK=np.loadtxt('Opt401.txt')
OO=np.loadtxt('Opt403.txt')
NN=np.loadtxt('Opt520.txt')
MM=np.loadtxt('Opt521.txt')
LL=np.loadtxt('Opt522.txt')

# Create an empty list to store the mean epsilon values
Mean_eps=[]
for i in range(12):
    Mean_eps.append(AA[1])
    Mean_eps.append(BB[1])
    Mean_eps.append(CC[1])
    Mean_eps.append(DD[1])
    Mean_eps.append(EE[1])
    Mean_eps.append(FF[1])
    Mean_eps.append(GG[1])
    Mean_eps.append(KK[1])
    Mean_eps.append(OO[1])
    Mean_eps.append(NN[1])
    Mean_eps.append(MM[1])
    Mean_eps.append(LL[1])

# Plot the histogram of the mean epsilon values
plt.hist(Mean_eps)
# Calculate the mean of the collected epsilon values
mean_value = np.mean(Mean_eps)
# Calculate standard deviation as 33% of the mean value
STD = mean_value * 0.33
# Plot histogram with 20 bins and normalized density
plt.hist(Mean_eps, bins=20, density=True, alpha=0.5, label='Histogram')


# Plot Gaussian distribution
# Define x values for plotting the Gaussian curve based on distances
x = np.linspace(min(distances), max(distances), 1000)
# Calculate the Gaussian (normal distribution) values with mean and standard deviation
y = gaussian(x, mean_value, STD)
plt.plot(x, y, color='red', label='Gaussian Distribution')


# Add labels and legend
plt.title('Histogram and Gaussian Distribution: NT')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.axvline(mean_value + STD, color='green', linestyle='dashed', linewidth=2, label='Mean + STD')
plt.axvline(mean_value - STD, color='green', linestyle='dashed', linewidth=2, label='Mean - STD')
plt.text(mean_value, max(y), f'Mean: {mean_value:.10f}', color='blue', fontsize=12, ha='center')

# Show plot
plt.grid(True)
plt.savefig('Histogram and Gaussian Distribution-TTX.png',dpi=1000)
plt.show()

# Open a file to save the optimal epsilon value (mean_value)
file=open('Optimal_eps_TTX.txt','w')
txt=str(mean_value)+'\n'
file.write(txt)
file.close() 

