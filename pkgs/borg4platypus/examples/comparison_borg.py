from platypus import *
import matplotlib.pyplot as plt
from borg4platypus import ExternalBorgC
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    # setup the experiment
    problem = DTLZ2(2)
        
    algorithms = [(ExternalBorgC, {"epsilons":0.01}, "borg_serial"),
                  (ExternalBorgC, {"epsilons":0.01, "mpirun":"-np 2"}, "borg_ms"),
                  NSGAII,
                  ]
    
    # run the experiment using Python 3's concurrent futures for parallel evaluation
    results = experiment(algorithms, problem, seeds=1, nfe=1000)

    # display the results
    for algorithm in six.iterkeys(results):
        result = results[algorithm]["DTLZ2"][0]
        plt.scatter(*[[s.objectives[i] for s in result] for i in range(2)], label=algorithm)
    plt.legend()
    plt.show()
