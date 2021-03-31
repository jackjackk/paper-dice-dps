from platypus import NSGAII, Problem, Real
from borg4platypus import SerialBorgC
import matplotlib.pylab as plt
import logging
import numpy as np

plt.interactive(True)

logging.basicConfig(level=logging.DEBUG)


def belegundu(vars):
    x = vars[0]
    y = vars[1]
    return ([-2 * x + y, 2 * x + y],
            [-x + y - 1, x + y - 7])


def ubelegundu(vars):
    x, y = vars
    return [-2 * x + y, 2 * x + y]


problem = Problem(2, 2, 2)
problem.types[:] = [Real(0, 5), Real(0, 3)]
problem.constraints[:] = "<=0"
problem.function = belegundu

algorithm = SerialBorgC(problem, 0.01)
#algorithm = NSGAII(problem)
algorithm.run(10000)

# Unconstrained
uproblem = Problem(2, 2, 0)
uproblem.types[:] = [Real(0, 5), Real(0, 3)]
uproblem.function = ubelegundu

plist = list(plt.rcParams['axes.prop_cycle'])

# Show constraints in solution space
def a2xy(a):
    xy = np.r_[[s.variables[:] for s in a.result]]
    return xy[:, 0], xy[:, 1]


x, y = a2xy(algorithm)

ux, uy = a2xy(ualgorithm)


for t in problem.types:
    xy.append(np.linspace(t.min_value, t.max_value))

import pandas as pd

algos = []
for p in [uproblem, problem]:
    algos.append([SerialBorgC(p, 0.01, pbar=False), NSGAII(p)])
    for a in algos[-1]:
        a.run(10000)

# Variable space
## Scatter
axs = plt.subplots(1,2, sharey=True)[1]
t=problem.types[0];xlin=np.linspace(t.min_value, t.max_value)
for ax, a2, lab in zip(axs,algos,['Unconstrained','Constrained']):
    for a in a2:
        df=pd.DataFrame([[s.variables[i] for s in a.result] for i in range(2)]).T
        ax.scatter(df[0],df[1])
    ax.set_title(lab)
    if lab=='Constrained':
        ax.plot(xlin, 1 + xlin, color='k')
        ax.plot(xlin, 7 - xlin, color='k')

## Heatmap
for o, ax in zip(ubelegundu(np.meshgrid(*xy)), plt.subplots(1,2)[1]):
    ax.imshow(o, origin='lower', extent=[0, xy[0][-1], 0, xy[1][-1]])
    for a in algos:
        ax.scatter(*[[s.variables[i] for s in a.result] for i in range(2)])
    
fig, ax = plt.subplots(1, 1)
ax.plot(xlin, 1 + xlin)
ax.plot(xlin, 7 - xlin)
ax.scatter(x, y)
ax.scatter(ux, uy)
# display the results
plt.scatter(*[[s.objectives[i] for s in algorithm.result] for i in range(2)])
plt.show()
