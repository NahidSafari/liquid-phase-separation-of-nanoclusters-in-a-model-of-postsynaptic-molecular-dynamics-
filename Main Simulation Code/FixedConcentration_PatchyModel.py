#!/usr/bin/env python
# coding: utf-8

#**************************************************************************************************
# ***     Simulation of a coarse-grained patchy model under fixed-concentration boundary condition using PyRID simulator    ***
# **************************************************************************************************/
#Copyright Nahid Safari ***


import pyrid as prd
import numpy as np
import sys 
import matplotlib.pyplot as plt


#-----------------------------------------------------
# Set Parameters
#-----------------------------------------------------

exp =  '4_sites' 
phase='fixed'
ia_id = int(sys.argv[1])


# File name and path
if exp == '4_sites': 
    interaction_strength =[0.0001,0.0002,0.0005,0.001,0.002,0.005,0.007,0.008,0.01,0.015,0.02]
    interaction_strength=np.array(interaction_strength)
    file_path='4_sites/Files/'
    fig_path = '4_sites/Figures/'
    if phase == 'fixed':
        file_name = '4_sites_'+phase+'_'+str(interaction_strength[ia_id])






#Simulation properties and System physical properties
nsteps = 1e8
stride = int(nsteps/1000)
obs_stride = int(nsteps/1000)
box_lengths = [74.0,74.0,74.0]
Temp=179.71
eta=1e-21
dt = 0.0025   # Think about it


#-----------------------------------------------------
# Initialize System
#-----------------------------------------------------

Simulation = prd.Simulation(box_lengths = box_lengths,
                            dt = dt,
                            Temp = Temp,
                            eta = eta,
                            stride = stride,
                            write_trajectory = True,
                            file_path = file_path,
                            file_name = file_name,
                            fig_path = fig_path,
                            boundary_condition = 'fixed concentration',
                            nsteps = nsteps,# fixed concentration
                            seed = 0,
                            length_unit = 'nanometer',
                            time_unit = 'ns')


#-----------------------------------------------------
# Add Checkpoints
#-----------------------------------------------------

Simulation.add_checkpoints(1000, "4_sites/checkpoints/", 1) 


#-----------------------------------------------------
# Define Particle Types
#-----------------------------------------------------

Simulation.register_particle_type('Core_1', 2.5)
Simulation.register_particle_type('Patch_1', 0.0)


#-----------------------------------------------------
# Add Global repulsive Pair Interactions
#-----------------------------------------------------

lr = 50
la = 49
EpsR = Simulation.System.kbt/1.5
Simulation.add_interaction('PHS', 'Core_1', 'Core_1', {'EpsR':EpsR, 'lr':lr, 'la':la})


#-----------------------------------------------------
# Add Pair Binding Reaction
#-----------------------------------------------------

eps_csw = 17.0
sigma = 2.5*2
alpha =  sigma*0.01 # 0.005 #
rw = 0.12*sigma
Simulation.add_bp_reaction('bind', ['Patch_1', 'Patch_1'], ['Patch_1', 'Patch_1'], 100/dt, 1.75*rw, 'CSW', {'rw':rw, 'eps_csw':eps_csw, 'alpha':alpha})

#-----------------------------------------------------
# Define Molecule Structure
#-----------------------------------------------------

A_pos = prd.distribute_surf.evenly_on_sphere(4,2.5)
A_types = np.array(['Core_1','Patch_1','Patch_1', 'Patch_1', 'Patch_1'], dtype = np.dtype('U20'))


#-----------------------------------------------------
# Register Molecules
#-----------------------------------------------------


Simulation.register_molecule_type('A', A_pos, A_types)

D_tt, D_rr = prd.diffusion_tensor(Simulation, 'A')
Simulation.set_diffusion_tensor('A', D_tt, D_rr)
prd.plot.plot_mobility_matrix('A', Simulation, save_fig = True, show = True)

#-----------------------------------------------------
# Load Molecules// Or distribute them
#-----------------------------------------------------

name= '4_sites_DC_'+str(eps_csw)
Simulation.load_checkpoint(name, 0, directory = 'checkpoints/')

#-----------------------------------------------------
# Set the boundary concentration parameters
#-----------------------------------------------------

concentration=interaction_strength[ia_id]
Simulation.fixed_concentration_at_boundary('A', concentration, 'Box', 'Volume')

#-----------------------------------------------------
# Add Observables
#-----------------------------------------------------

Simulation.observe('Number', ['A'], obs_stride = obs_stride)

#-----------------------------------------------------
# Start the Simulation
#-----------------------------------------------------


Simulation.run(progress_stride = 1000, out_linebreak = False)

Simulation.print_timer()
Evaluation = prd.Evaluation()
Evaluation.load_file(file_name)



Evaluation.plot_observable('Number', ['A'], save_fig = True)
plt.axhline(1000, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)

plt.savefig('Figures//Fixed_Concentration_Number.png', bbox_inches="tight", dpi = 300)
prd.plot.plot_scene(Simulation, save_fig = True)
prd.plot.plot_concentration_profile(Simulation, axis = 0, save_fig = True)





