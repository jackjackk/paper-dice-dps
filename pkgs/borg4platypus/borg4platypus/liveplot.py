# inspired by https://www.youtube.com/watch?v=ZmYPzESC5YY
import os
import collections
import glob
from collections import defaultdict
from functools import partial
import logging
import matplotlib.pyplot as plt
import sys

import dill
from matplotlib import animation
import matplotlib as mpl
import pandas as pd
from matplotlib.animation import FuncAnimation
import numpy as np
from platypus import EpsilonBoxArchive, Hypervolume, Archive
import time
import platypus as pts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('liveplot')

qs = defaultdict(lambda: collections.deque(5*[None], 5))
last_indices = defaultdict(lambda: collections.deque(2*[None], 2))
last_hyps = defaultdict(lambda: collections.deque(2*[None], 2))
last_index = None
colors = [list(np.array(list(bytes.fromhex(x['color'][1:])))/255) for x in mpl.rcParams['axes.prop_cycle']]

Solution = lambda *args: pd.Series(args, index=['variables', 'objectives', 'constraints',
                                               'constraint_violation', 'problem', 'evaluated',
                                               'normalized_objectives'])

reference_set = EpsilonBoxArchive(0.01)

#hyp = Hypervolume(reference_set)

solution_set = defaultdict(lambda: Archive())
solution_set = defaultdict(list)


def animate(i,
            datadir, # file path of the Borg runtime file to plot
            axs,
            problem,
            hyp
            ):
    time_start = time.perf_counter()
    time_hyp = 0
    for df, col in zip(glob.glob(os.path.join(datadir, '*.csv')), colors):
        #try:
            q = qs[df]
            last_indices_df = last_indices[df]
            last_index = last_indices_df[-1]
            solution_set = []
            try:
                y = pd.read_csv(df, comment='/').reset_index().set_index(['NFE','index']) # [['obj0', 'obj1']]
                y.index.levels[0][-1]
            except:
                continue
            if last_index != y.index.levels[0][-1]:
                last_index = y.index.levels[0][-1]
                for i, hscat in enumerate(q):
                    if hscat is None:
                        continue
                    hscat.set_facecolors(col + [1. / ((len(q)-i) + 1), ])
                    hscat.set_edgecolors(col + [1. / ((len(q)-i) + 1), ])
                curr_hscat = axs[0].scatter(y.loc[last_index, 'obj0'], y.loc[last_index, 'obj1'], color=col)
                vcols = [x for x in y.columns.values if 'dv'==x[:2]]
                ocols = [x for x in y.columns.values if 'obj'==x[:3]]
                for idx, sol_series in y.loc[last_index].iterrows():
                    sol = Solution(sol_series[vcols].values,
                                   sol_series[ocols].values,
                                   [],
                                   0,
                                   problem,
                                   True,
                                   [])
                    #reference_set.add(sol)
                    solution_set.append(sol)
                time_checkpoint = time.perf_counter()
                last_hyp = hyp.calculate(solution_set)
                time_hyp += (time.perf_counter()-time_checkpoint)
                last_hyps[df].append(last_hyp)
                last_indices_df.append(last_index)
                if last_indices_df[0] is not None:
                    axs[1].plot(last_indices_df, last_hyps[df], color=col)
                else:
                    axs[1].scatter(last_index, last_hyp)
                if q[0] is not None:
                    q[0].remove()
                q.append(curr_hscat)
        #except:
        #    pass
    logger.debug('{timeelap} elapsed ({time_hyp} for hyp)'.format(timeelap=time.perf_counter()-time_start,time_hyp=time_hyp))



if __name__ == '__main__':
    os.chdir(sys.argv[1])
    pidfile = 'liveplot.pid'
    logger.info('Dir = {cwd}'.format(cwd=os.getcwd()))
    if os.path.isfile(pidfile):
        logger.info('Liveplot already launched')
        sys.exit(1)
    with open(pidfile, 'w') as f:
        f.write(str(os.getpid()))
    with open(os.path.join(sys.argv[1], 'liveplot_problem.dat'), 'rb') as f:
        problem = dill.load(f)
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[1].set_xlim([0, int(sys.argv[3])])
    #for _ in range(1000): reference_set.add(problem.random())
    oref = pd.read_csv(os.path.join(sys.argv[1], 'liveplot_reference.dat'), index_col=0)
    omin = oref.loc['omin']
    omax = oref.loc['omax']
    hmin = np.min([np.min([omin, omax], 0)-abs(omax-omin)*10, np.zeros(len(omin))], 0)
    hmax = np.max([np.max([omin, omax], 0)+abs(omax-omin)*10, np.zeros(len(omin))], 0)
    logger.info('For hypervolume, using min={hmin}, max={hmax}'.format(hmin=hmin.tolist(),hmax=hmax.tolist()))
    hyp = Hypervolume(minimum=hmin, maximum=hmax)
    panimate = partial(animate, datadir=sys.argv[1], axs=axs, problem=problem, hyp=hyp)
    ani = animation.FuncAnimation(fig, panimate, repeat_delay=int(sys.argv[2])*1000, repeat=True)
    plt.show(block=True)
    os.unlink(pidfile)



