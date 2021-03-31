from platypus import NSGAII, Problem, Real
from borg4platypus import SerialBorgC
import matplotlib.pylab as plt
import logging
import numpy as np
import pandas as pd
plt.interactive(True)
logging.basicConfig(level=logging.DEBUG)

# 

def binh_korn(vars):
    '''https://en.wikipedia.org/wiki/Test_functions_for_optimization'''
    x = vars[0]
    y = vars[1]
    f1 = 4*x**2 + 4*(y**2)
    f2 = (x-5)**2 + (y-5)**2
    g1 = (x-5)**2 + y**2 - 16
    g2 = 7.7 - (x-8)**2 - (y+3)**2
    if (g1 > 0) or (g2 > 0):
        g1=g1
        pass
    return ([f1, f2], [g1, g2])


def ubinh_korn(vars):
    f, g = binh_korn(vars)
    return f


probs = []
for f, nc in zip([binh_korn, ubinh_korn], [2,0]):
    problem = Problem(2, 2, nc)
    problem.types[:] = [Real(0, 5), Real(0, 3)]
    problem.constraints[:] = "<=0"
    problem.function = f
    probs.append(problem)


algos = []
for p in probs:
    algos.append([SerialBorgC(p, 0.01, pbar=False), NSGAII(p)])
    for a in algos[-1]:
        a.run(10000)


plist = list(plt.rcParams['axes.prop_cycle'])


# Variable space
## Scatter
axs = plt.subplots(1,2, sharey=True)[1]
t=problem.types[0];xlin=np.linspace(t.min_value, t.max_value)
for ax, a2, lab in zip(axs,algos,['Constrained','Unconstrained']):
    for a in a2:
        df=pd.DataFrame([[s.variables[i] for s in a.result] for i in range(2)]).T
        ax.scatter(df[0],df[1])
    ax.set_title(lab)
    if lab=='Constrained':
        ax.fill_between(xlin, np.zeros(len(xlin)), np.sqrt(16-(xlin-5)**2), alpha=0.5, color='k')
        ax.fill_between(xlin, np.minimum(3, np.maximum(-3+np.sqrt(np.abs(7.7-(xlin-8)**2)),0)), 3*np.ones(len(xlin)), alpha=0.5, color='k')

# Objectives
axs = plt.subplots(1,2, sharey=True)[1]
for ax, a2, lab in zip(axs,algos,['Unconstrained','Constrained']):
    for a in a2:
        df=pd.DataFrame([[s.objectives[i] for s in a.result] for i in range(2)]).T
        ax.scatter(df[0],df[1])
    ax.set_title(lab)
