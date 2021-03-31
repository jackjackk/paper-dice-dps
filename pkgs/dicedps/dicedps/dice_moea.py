"""
Executable interface for running dicedps experiments.
Use :ref:`qsub_dice` for running on cluster.
"""

from pathos.multiprocessing import ProcessPool
from platypus import experiment, PoolEvaluator
from borg4platypus import SerialBorgC, ExternalBorgC

import numpy as np

from dicedps.algos_tqdm import NSGAIIp
from dicedps.objectives import oset2vout, obj2eps
from dicedps.dice_helper import get_uncertain_dicesim, args2dice, check_signal_bounds

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dice_borg')

from dicedps.qsub_dice import get_parsed_args, args2name


def main_dice_moea(args=None):
    """
    Parse options, build DICE, run BORG.

    :return: Nothing.
    """

    # Parse options
    args = get_parsed_args(args)
    #args = get_parsed_args(['-w 10'])



    # Build DICE
    vout = oset2vout[args.obj]
    dc = args2dice(args)
    if not args.miu.startswith('time'):
        assert check_signal_bounds(args)
    pdice = dc.asproblem()

    logger.info('Run preliminary run')
    miu0 = 0.5 * np.ones(len(dc.get_bounds()))
    dc.run(miu0, track=True)

    # Run algo:
    name = args2name(args)
    algo_args = dict(log_frequency=max(10, args.nfe//20), runtimedir=os.path.abspath(args.outpath),
                     tempdir=os.path.abspath(os.path.join(args.outpath, args2name(args))))
    if 'borg' in args.algo:
        epsilons = [obj2eps[o] for o in vout]
        algo_args.update(dict(epsilons=epsilons, liveplot=False, pbar=True,
                         name=name,backend=args.algo))
        if args.seeds == 1:
            algo_args.update(dict(seed=args.iseed))
        if args.procs == 1:
            algo_class = SerialBorgC
            algo_args.pop('tempdir')
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
        algo = algo_class(pdice, do_postprocess=False, do_solve=(not args.dryrun), **algo_args)
        algo.run(args.nfe)
        #raw_results = [algo.result,]

    logger.info('Done')


if __name__ == '__main__':
    main_dice_moea()





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
