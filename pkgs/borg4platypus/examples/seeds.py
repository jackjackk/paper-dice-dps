from pathos.multiprocessing import ProcessPool, freeze_support
from platypus import experiment, DTLZ2, PoolEvaluator

from borg4platypus import ExternalBorgC, SerialBorgC, tqdm

import logging
logging.basicConfig(level=logging.INFO)

problem = DTLZ2()

import glob, os
for f in glob.glob('seeds_seed*csv'):
    os.unlink(f)

algorithms = [(SerialBorgC, {"epsilons":0.01,
                               'log_frequency':1000,
                               'name':'seeds', 'liveplot':True,
                               'pbar':False}, "borg_serial")]

peval = PoolEvaluator(ProcessPool(ncpus=4))
results = experiment(algorithms, problem, seeds=10, nfe=100000, evaluator=peval)
