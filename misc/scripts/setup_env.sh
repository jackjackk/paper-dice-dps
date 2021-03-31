#!/bin/bash

# Load modules
module purge
module load python/3.6.3-anaconda5.0.1 cmake/3.8.2 gcc/5.3.1 netcdf/4.4.0 openmpi/1.10.1 openjdk/1.8.0

# Activate Anaconda environment
source activate dicedps2

# Assume the script is included from root directory!
LD_LIBRARY_PATH=$(pwd)/pkgs/borg/build/lib:${LD_LIBRARY_PATH}
