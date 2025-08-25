import numpy as np
import matplotlib.pyplot as plt

# Load data from a text file into a NumPy array
data=np.loadtxt('percentages_DMSO.txt')

# Define column labels for the dataset, where each entry corresponds to a specific data type or category
Col = [r'$\epsilon$', r'$N_{min}$', 'Percentage of MC', 
       'Percentage of SC', 'Percentage of LC']
print(Col)

# Pre-calculate mean and standard deviation for all columns
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

# Loop through each column to create a boxplot
for i in range(data.shape[1]):
    plt.figure(figsize=(6, 4), dpi=300)  # High DPI for better image quality
    plt.boxplot(data[:, i], 
                patch_artist=True, 
                boxprops=dict(facecolor='white', linewidth=2),  # Thicker border
                whiskerprops=dict(linewidth=2),  # Thicker whiskers
                capprops=dict(linewidth=2),  # Thicker caps
                medianprops=dict(linewidth=2))  # Thicker median line
    # Create boxplot with a customized color
#     plt.boxplot(data[:, i], patch_artist=True, boxprops=dict(facecolor='white'))
    
    # Set the title of each plot with the corresponding column name
#     plt.title(f'Boxplot of {Col[i]}', fontsize=15)
    
    # Add a grid to the plot for easier interpretation
    plt.grid(True, linestyle='--', alpha=0.6)

    # Calculate and display mean and standard deviation in a textbox
    stats_text = f'Mean: {means[i]:.8f}\nStd: {stds[i]:.8f}'
    plt.text(0.61, 0.85, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    # Set labels for the axes
#     plt.xlabel('Data Columns', fontsize=15)
    plt.ylabel(str(Col[i]), fontsize=20)
    plt.xticks([])
    plt.yticks(fontsize=15)
    # Label the x-axis with the corresponding column name
#     plt.xticks([1], [Col[i]], fontsize=15)

    # Save the plot as a PNG file with high resolution
    plt.savefig(f'boxplot_column_{i+1}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    # Close the current plot to avoid memory overload
    plt.close()

