"""
Executable interface for running dicedps experiments on cluster.
"""
import re
import sys
import argparse
import os
from collections import OrderedDict
import pandas as pd

from dicedps.doeclim_moea import get_parsed_args, args2name
from dicedps.objectives import oset2vout
import numpy as np

def main_qsub_doeclim():
    """
    1. Parse args
    2. Write PBS file using args
    3. Submit PBS job
    """
    args = get_parsed_args()
    name = args2name(args)
    nodes = int(np.ceil(args.procs/20))
    ppn = min(args.procs, 20)
    pbsfile_contents = f"""#!/bin/bash
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l walltime={args.hours}:00:00
#PBS -l pmem=2gb
#PBS -A {args.queue}
#PBS -j oe
#PBS -o {name}.log

# Get started
echo " "
echo "Job started on `hostname` at `date`"
echo " "

# Get environment 
module purge
. ~/.bash_python
. ~/.bash_borg
module use /gpfs/group/kzk10/default/sw/modules
module load gcc
module load openmpi/1.10.1

# Go to the correct place
cd $PBS_O_WORKDIR

# Clean up
rm -fv {name}_*.csv

# Run the job itself
python3 --version
python3 -m dicedps.doeclim_moea {" ".join(sys.argv[1:])} -D
mpirun -np {args.procs} -machinefile $PBS_NODEFILE $(which python3) $PBS_O_WORKDIR/{name}/runscript.py

# Finish up
echo " "
echo "Job Ended at `date`"
echo " "
"""
    with open(name + '.pbs', 'w') as f:
        f.write(pbsfile_contents)
    os.system(f'qsub {name}.pbs')


if __name__ == '__main__':
    main_qsub_doeclim()