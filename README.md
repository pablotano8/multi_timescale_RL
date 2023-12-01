# Multi-timescale reinforcement learning in the brain

## General instructions and expected outputs
The names of the files start with the figure that they reproduce. Some scripts reproduce multiple figures (e.g. Fig_2_c_d_e). In these cases there are commented instructions in the script that indicate how to change the parameters to reproduce the desired figure.

## Required packages
Required packages are listed in requirements.txt. We recommend creating a conda environment with all the necessary packages before running the scripts.

## Approximate execution times
Approximate execution times on an Apple MacbookPro M2 Max with 32GB of RAM (MacOS 13.2.1):

- **Fig_2_c_d_e and Ext_Fig_1:** 6 minutes for each set of discounts 
- **Fig_2_f_myopic_mdp:** 1 minute to evaluate performance of multi-timescale agents, 4 minutes to produce figure
- **Fig_2_g_train_lunar_multi_gamma:** 9 minutes to train agent for 50000 frames
- **Fig_2_g_lunar_q_accuracy:** 5 minutes per network (50 minutes for the 10 networks in the script)
- **Ext_fig_2:** a few seconds
- **Ext_Fig_3_myopic_bias_maze:** 31 minutes


## System Requirements
### Hardware requirements
The scripts are written for CPU, but execution times could improve if adapted to GPU. The code requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
The code has been tested on the following systems:
+ macOS: Mojave (13.2.1)

## License
This project is covered under the MIT License (see LICENSE file).
