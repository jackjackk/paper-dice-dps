from borg4platypus import SerialBorgC
from platypus import Problem, Real, EpsilonBoxArchive, Solution
import numpy as np


def twoquad(vars):
    x1 = vars[0]
    x2 = vars[1]
    f1 = 5*pow(x1-0.1, 2)+pow(x2-0.1, 2)
    f2 = pow(x1 - 0.9, 2)+pow(x2-0.9, 2)
    return [f1, f2]

def twoquadconstr(vars):
    return twoquad(vars), [vars[0]-0.4,vars[1]-0.4]

problemc = Problem(2, 2, 2)
problemc.types[:] = [Real(0, 1), Real(0, 1)]
problemc.constraints[:] = "<=0"
problemc.function = twoquadconstr

algorithmc = SerialBorgC(problemc, 0.01)
algorithmc.run(100000)

def a2xy(a):
    xy = np.r_[[s.variables[:] for s in a.result]]
    return xy[:,0], xy[:,1]
x, y = a2xy(algorithmc)


