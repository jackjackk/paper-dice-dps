"""
Executable interface for running dicedps experiments.
Use :ref:`qsub_dice` for running on cluster.
"""
import argparse

from pathos.multiprocessing import ProcessPool
from platypus import experiment, PoolEvaluator
from borg4platypus import SerialBorgC, ExternalBorgC

import numpy as np

from paradoeclim.calibration import DoeclimCalibLikelihood
from .algos_tqdm import NSGAIIp
from .uncertainties import ulist3, sample_combinatorial
from .objectives import oset2vout, obj2eps
from .dice_helper import get_uncertain_dicesim, args2dice, check_signal_bounds
from .dpsrules import MiuProportionalController, MiuPolyController, MiuKlausController, MiuRBFController, \
    MiuTemporalController, dpslab2signals, args2dpsclass

from functools import partial
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('doeclim_borg')


def get_parsed_args(args=None):
    parser = argparse.ArgumentParser(description='Calibrate Doeclim w/ Borg')

    # Solver
    parser.add_argument('-a', '--algo', dest='algo', type=str, action='store',
                        default='borg', choices={'borg','nsga'}, help='MOEA Algorithm')
    parser.add_argument('-n', '--nfe', dest='nfe', type=int, action='store',
                        default=1000, help='Number of function evaluations')
    parser.add_argument('-H', '--hours', dest='hours', type=int, action='store',
                        default=23, help='Number of hours for PBS job')
    parser.add_argument('-D', '--dryrun', dest='dryrun', action='store_true',
                        default=False, help='Dry run')


    # Borg
    parser.add_argument('-p', '--procs', dest='procs', action='store', type=int,
                        default=1, help='Number of cores for MPI master-slave runs')
    parser.add_argument('-i', '--islands', dest='islands', action='store', type=int,
                        default=1, choices={1,}, help='Number of islands for Borg')

    # Doeclim setup, Inputs & outputs
    parser.add_argument('-c', '--cs', dest='cs', action='store', type=float,
                        default=3.1, help='Given climate sensitivity')

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

    if isinstance(args, str):
        args=args.split(' ')

    return parser.parse_args(args=args)


def args2name(args):
    return f'doeclim_calib_cs{args.cs:.2f}_i{args.islands}p{args.procs}_nfe{args.nfe}_s{args.iseed}'


def main_doeclim_moea(args=None):

    # Parse options
    args = get_parsed_args(args)

    # Build Doeclim
    vout = ['NEG_LOGLIKELIHOOD']
    dc = DoeclimCalibLikelihood(vin=['kappa', 'alpha',
                              'temp_wsigma', 'heat_wsigma',
                              'temp_rho', 'heat_rho', 'temp0', 'heat0'], setup={'t2co':args.cs})
    prob = dc.asproblem()

    # Run algo:
    name = args2name(args)
    algo_args = dict(log_frequency=max(10, args.nfe//20),
                     tempdir=os.path.abspath(args2name(args)))
    if args.algo == 'borg':
        epsilons = [1e-3]
        algo_args.update(dict(epsilons=epsilons, liveplot=False, pbar=True,
                         name=name))
        if args.seeds == 1:
            algo_args.update(dict(seed=args.iseed))
        if args.procs == 1:
            algo_class = SerialBorgC
        else:
            algo_args['np'] = args.procs
            if args.procs>10:
                algo_args['pbar'] = False
            algo_args['islands'] = args.islands
            algo_class = ExternalBorgC
    elif args.algo == 'nsga':
        algo_class = NSGAIIp
    else:
        assert False, f'"{args.algo}" not supported'

    if args.seeds > 1:
        algo = (algo_class, algo_args, args.algo)
        peval = PoolEvaluator(ProcessPool(ncpus=args.seeds))
        #raw_results = (experiment(algo, pdice, seeds=args.seeds, nfe=args.nfe, evaluator=peval))[args.algo]['Problem']
    else:
        algo = algo_class(prob, do_postprocess=False, do_solve=(not args.dryrun), **algo_args)
        algo.run(args.nfe)
        #raw_results = [algo.result,]

    logger.info('Done')


if __name__ == '__main__':
    main_doeclim_moea()





"""
results = []
assert len(dc._vin) == 1, 'More than one input array not supported'
for r in raw_results:
    ds = rh.DataSet()
    for sol in r:
        ds.append({dc._vin[0]:list(sol.variables), **dict(zip(mdice.responses.keys(), sol.objectives))})
    results.append(ds)

# Save results
#with open(f'results_{name}.dat','wb') as f:
#    dill.dump((dc, mdice, results), f)

pplot([bau, dc.d], 'S')
dcp = dc.asproblem()
max_nfe = 1000
log_freq = 100
epss = 0.01

import os

# algo = ExternalBorgC(dcp, epsilons=epss,
#                           log_frequency=log_freq,
#                           name='dctest', mpirun='-np 2')
algo = SerialBorgC(dcp, epsilons=epss,
                          log_frequency=log_freq,
                          liveplot=True, pbar=True,
                          name='dctest',seed=1)
algo.run(max_nfe)
"""