import sys
from platypus import DTLZ2
from platypus.core import EpsilonBoxArchive, Solution
from platypus.indicators import Hypervolume
from borg4platypus import ExternalBorgC, SerialBorgC, MpiBorgC
import logging

logging.basicConfig(level=logging.DEBUG)

# define the problem definition
problem = DTLZ2()

# instantiate the optimization algorithm
max_nfe = 1000000
log_freq = int(max_nfe/10)
epss = 0.01
seed = 1

algorithm = ExternalBorgC(problem, epss, np=3, tempdir='test_mpi', seed=seed,
                          log_frequency=log_freq, pbar=True, name='dtlz2')

# optimize the problem using max_nfe function evaluations
algorithm.run(max_nfe)

# display hypervolume wrt set of random solutions
#reference_set = EpsilonBoxArchive(epss)
#for _ in range(1000): reference_set.add(problem.random())
#print("Hypervolume:", Hypervolume(reference_set).calculate(algorithm.result))
