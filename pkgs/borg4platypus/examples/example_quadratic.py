from borg4platypus import SerialBorgC
from platypus import Problem, Real, EpsilonBoxArchive, Solution
import numpy as np
import matplotlib.pylab as plt
plt.interactive(True)
import seaborn as sb


def twoquad(vars):
    x1 = vars[0]
    x2 = vars[1]
    f1 = 5*pow(x1-0.1, 2)+pow(x2-0.1, 2)
    f2 = pow(x1 - 0.9, 2)+pow(x2-0.9, 2)
    return [f1, f2]

def twoquadconstr(vars):
    return twoquad(vars), [vars[0]-0.4,vars[1]-0.4]


x1=np.linspace(0, 1, 100)   # each variable is instantiated N=10 times
x2=x1.copy()
Z=twoquad([x1, x2])
fig, axs = plt.subplots(1,2)
for i in range(2):
    sb.heatmap(Z[i], ax=axs[i])

problem = Problem(2, 2, 0)
problem.types[:] = [Real(0, 1), Real(0, 1)]
problem.constraints[:] = "<=0"
problem.function = twoquad #constr

problemc = Problem(2, 2, 2)
problemc.types[:] = [Real(0, 1), Real(0, 1)]
problemc.constraints[:] = ">=0"
problemc.function = twoquadconstr

gridset = EpsilonBoxArchive(0.01)
xx1, xx2 = np.meshgrid(x1, x2)
for xpoint in np.nditer([xx1,xx2]):
    s = Solution(problem)
    s.variables[:] = xpoint
    s.evaluate()
    gridset.add(s)

gridset.add()
algorithm = SerialBorgC(problem, 0.01)
#algorithm = NSGAII(problem)
algorithm.run(1000000)

algorithmc = SerialBorgC(problemc, 0.01)
algorithmc.run(1000000)

def a2xy(a):
    xy = np.r_[[s.variables[:] for s in a.result]]
    return xy[:,0], xy[:,1]
x, y = a2xy(algorithm)

xmin, xmax = x.min(), x.max()
xlin = np.linspace(xmin, xmax)

cols = sb.color_palette()
fig, ax = plt.subplots(1,1)
ax.scatter(x, y, color=cols[1])
ax.scatter(*a2xy(algorithmc), color=cols[5])
ax.scatter(*[[s.variables[i] for s in gridset] for i in range(2)])

xs = np.sort(x)
ax.plot(xs, 0.1*(-224*xs+21.6)/(-24*xs+1.6))

