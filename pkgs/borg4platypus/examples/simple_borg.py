import sys
from platypus.problems import DTLZ2
from platypus.core import EpsilonBoxArchive, Solution
from platypus.indicators import Hypervolume
from borg4platypus import ExternalBorgC, SerialBorgC, MpiBorgC
import logging

logging.basicConfig(level=logging.DEBUG)

# define the problem definition
problem = DTLZ2()

# instantiate the optimization algorithm
max_nfe = 100000
log_freq = int(max_nfe/100)
epss = 0.01

try:
    seed=int(sys.argv[1])
except:
    seed=1

algorithm = SerialBorgC(problem, epss, seed=seed,
                          log_frequency=log_freq,
                          name='dtlz2', liveplot=False, pbar=True,
                          #mpirun='-np 1', tempdir='test',
                     )

# optimize the problem using max_nfe function evaluations
algorithm.run(max_nfe)

# display hypervolume wrt set of random solutions
reference_set = EpsilonBoxArchive(epss)
for _ in range(1000): reference_set.add(problem.random())
print("Hypervolume:", Hypervolume(reference_set).calculate(algorithm.result))
