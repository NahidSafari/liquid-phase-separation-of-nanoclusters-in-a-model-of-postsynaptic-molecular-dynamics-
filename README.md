# Controlling the Formation of Multiple Condensates in the Synapse

## Getting started
Computational models and MINFLUX microscopy data revealing how protein availability and binding affinities regulate the formation, number, and plasticity-driven remodeling of nanoscale synaptic nanoclusters.



## 📂 Repository Structure


```plaintext
├── 📂 Main Simulation Code/         # Core computational models for the formation of different phases
│   ├──  📄 FixedConcentration_PatchyModel.py                    ## Simulation of a coarse-grained patchy model under fixed-concentration boundary condition using PyRID simulator
│   ├──  📄 run_Simulation_Main.sh                               ## Slurm batch script for running FixedConcentration_PatchyModel.py on an HPC cluster
├── 📂 MINFLUX Data Analysis/        # Scripts for processing and analyzing MINFLUX data
│   ├──  📄 1_Error_using_average_eps_Each_Culture.py            ##  DBSCAN clustering Method, Error Estimation
│   ├──  📄 2_Error_using_average_eps_all_cultues.py             ##  DBSCAN clustering Method, Error Estimation
│   ├──  📄 3_Clustering_analysis_different__cultures_spine.py   ##  DBSCAN clustering Method, Error Estimation
│   ├──  📄 4_Boxplot_Visualization_Statistical_Summary.py       ##  Loads data, computes stats, and creates annotated boxplots saved as PNG files.
├── 📂 Analysis/                     # Additional analysis and figure generation scripts
│ ├── 📂 Clustering methods                # Contains scripts implementing various clustering algorithms
│   ├──  📄 Mean_Shift.py                                        ## MeanShift Clustering Method
│   ├──  📄 Clustering-connectivity.py                           ## Clustering Using Connectivity matrix and Graph Theory
│   ├──  📄 DBSCAN.py                                            ## DBSCAN Clustering Method using elbow method to determine eps
│ ├── 📂 Phase Diagram                     # Phase Diagram
│   ├──  📄 Phase_Diagram.py                                     ## Phase Diagram Plot 
│ ├── 📂 Various Quantities                # Calculating different Quantities in the system
│   ├──  📄 Criticality.py                                       ## Investigating Criticality
│   ├──  📄 Jaccard.py                                           ## Jaccard Index to explore the stability of Clusters over time 
│   ├──  📄 Number.py                                            ## Variation of the number of particles over time 
├── requirements.txt              # Python dependencies
└── LICENSE                       # License information

