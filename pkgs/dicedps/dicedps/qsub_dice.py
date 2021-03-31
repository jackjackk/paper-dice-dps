"""
Executable interface for running dicedps experiments on cluster.
"""
import re
import sys
import argparse
import os
from collections import OrderedDict
import pandas as pd

from dicedps.objectives import oset2vout
from dicedps.dpsrules import miu2dargs

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='Experiment w/ DICE & Borg')
    # Solver
    parser.add_argument('-a', '--algo', dest='algo', type=str, action='store',
                        default='borgpy', choices={'cyborg','borgpy','nsga'}, help='MOEA Algorithm')
    parser.add_argument('-n', '--nfe', dest='nfe', type=int, action='store',
                        default=1000, help='Number of function evaluations')
    parser.add_argument('-H', '--hours', dest='hours', type=int, action='store',
                        default=23, help='Number of hours for PBS job')
    parser.add_argument('-D', '--dryrun', dest='dryrun', action='store_true',
                        default=False, help='Dry run')
    parser.add_argument('-O', '--outpath', dest='outpath', action='store',
                        type=str, default='output/dicedps', help='output path')


    # Borg
    parser.add_argument('-p', '--procs', dest='procs', action='store', type=int,
                        default=1, help='Number of cores for MPI master-slave runs')
    parser.add_argument('-i', '--islands', dest='islands', action='store', type=int,
                        default=1, choices={1,}, help='Number of islands for Borg')

    # Dice setup, Inputs & outputs
    parser.add_argument('-m', '--miu', dest='miu', action='store',
                        default='TdT', help='MIU Control structure (e.g. time, T, TdT)')
    parser.add_argument('-r', '--rbfn', dest='rbfn', action='store',
                        type=int, default=4, help='Number of RBFs')
    parser.add_argument('-X', '--thres', dest='thres', action='store',
                        type=int, default=1, help='Include thresholding along with RBF')
    parser.add_argument('-B', '--maxrate', dest='maxrate', action='store',
                        type=int, default=4, help='Maximum rage')
    parser.add_argument('-T', '--miustep', dest='miustep', action='store',
                        type=int, default=1, help='Miu step update')
    parser.add_argument('-M', '--miuinterp', dest='miuinterp', action='store',
                        type=int, default=1, help='Miu step interpolation method')
    parser.add_argument('-o', '--obj', dest='obj', action='store',
                        choices=list(oset2vout.keys()),
                        default='simple2', help='Choice of objectives')
    parser.add_argument('-x', '--con', dest='con', action='store',
                        choices=['inertmax','inert95q','none'],
                        default='none', help='Choice of constraints')
    parser.add_argument('-R', '--reset', dest='reset', action='store_true',
                        default=False, help='Reset bau4bge.dat')
    parser.add_argument('-c', '--climate', dest='climate', action='store',
                        choices={'doeclim','dice'},
                        default='doeclim', help='Choose climate model')
    parser.add_argument('-C', '--climcalib', dest='climcalib', action='store',
                        #choices={'ka16','ka18','k18','c16'},
                        default='med', help='Choose climate model calibration or an .nc MCMC output file or an int index')
    parser.add_argument('-t', '--climt0', dest='climt0', action='store_true',
                        default=True, help='Include calibrated temp0')
    parser.add_argument('-d', '--damfunc', dest='damfunc', action='store', type=int,
                        default=1, help='Damage function specification')

    # Uncertainties
    parser.add_argument('-u', '--nunc', dest='nunc', action='store', type=str,
                        default='1', help='Type of uncertainties')
    parser.add_argument('-w', '--sows', dest='sows', type=str, action='store',
                        default='10', #choices={'1','3','5','10','100','1000','10000','100x10x10'},
                        help='Number of SOWs')
    parser.add_argument('-e', '--endyear', dest='endyear', type=int, action='store',
                        default=2250, help='End year')

    # Experiment
    parser.add_argument('-s', '--seeds', dest='seeds', type=int, action='store',
                        default=1, help='Number of seeds')
    parser.add_argument('-S', '--iseed', dest='iseed', type=int, action='store',
                        default=1, help='Seed number')
    parser.add_argument('-q', '--queue', dest='queue', type=str, action='store',
                        default='kzk10_a_g_sc_default', choices={'kzk10_a_g_sc_default','open'},
                        help='Account string')

    #parser.add_argument('-b', '--mpi', dest='mpi', action='store',
    #                    default='serial', choices={'nsga','serial','mpi'}, help='Type of solver')
    return parser


def get_parsed_args(args=None):
    parser = get_parser()

    if isinstance(args, str):
        args=args.split(' ')

    return parser.parse_args(args=args)


def args2miu(args):
    if args.miu.startswith('time'):
        miu = args.miu
    else:
        miu = f'rbf{args.miu}{args.rbfn}{args.thres}'
    return miu


def miu2arg(miu):
    """Convert miu label into args string."""
    return dargs2sargs(miu2dargs(miu))


def args2name(args):
    return f'u{args.nunc}w{args.sows}{args.climate}_m{args2miu(args)}_i{args.islands}p{args.procs}_nfe{args.nfe}_obj{args.obj}_c{args.con}_s{args.iseed}'


def name2dargs(pathname, save_miulab=True):
    name = os.path.basename(pathname)
    vlist = re.match(
               'u([^w]+)'
              'w([0-9]+)'
                '([^_]+)'
              '_m([^_]+)'
              '_i([^p]+)'
               'p([^_]+)'
            '_nfe([0-9]+)'
            '([^_]*)'
            '_obj([^_]+)'
              '_c([^_]+)'
              '_s([^_]+)'
           '_seed([^_]+)'
            '(_C([^_]+))?'
            '(_D([0-9]))?'
           '.*', name).groups()
    arglist = ['nunc','sows','climate','miulab','islands','procs','nfe','nfesuff','obj','con','seeds','iseed',None,'climcalib',None,'damfunc']
    dargs = {}
    for arg, v in zip(arglist, vlist):
        if (arg is None) or (v is None):
            continue
        if (not save_miulab) and (arg == 'miulab'):
            dargs.update(miu2dargs(v))
        elif arg == 'nfesuff':
            continue
        else:
            dargs[arg] = v
    dargs['endyear'] = 2250
    return dargs


def dargs2sargs(dargs):
    sargs = ''
    for k,v in dargs.items():
        sargs += f' --{k}={v}'
    return sargs.lstrip()


def fname2index(fname):
    return pd.Series(name2dargs(os.path.basename(fname)))


def main_qsub_dice():
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
ulimit -s 10240

# Get environment 


# Go to the correct place
cd $PBS_O_WORKDIR

source misc/scripts/setup_env.sh

# Clean up
rm -fv {args.outpath}/{name}_*.csv

# Run the job itself
python --version
python -m dicedps.dice_moea {" ".join(sys.argv[1:])} -D
cd {args.outpath}
mpirun -np {args.procs} -machinefile $PBS_NODEFILE $(which python) $PBS_O_WORKDIR/{args.outpath}/{name}/runscript.py

# Finish up
echo " "
echo "Job Ended at `date`"
echo " "
"""
    with open(name + '.pbs', 'w') as f:
        f.write(pbsfile_contents)
    os.system(f'qsub {name}.pbs')


if __name__ == '__main__':
    main_qsub_dice()
