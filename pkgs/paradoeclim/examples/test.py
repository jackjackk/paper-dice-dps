
import itertools

from doeclim import DoeclimCalib
from model import MODE_SIM, MODE_OPT, Time
import matplotlib.pylab as plt
import logging

logging.basicConfig(level=logging.INFO)


#dc = DoeclimCalib(name='doeclim_sim', mode=MODE_SIM, vin=['t2co','kappa','alpha'])

vin = ['t2co', 'kappa', 'alpha', 'dq10', 'dq20', 'delql0', 'delqo0', 'dpast20', 'dteaux10', 'dteaux20']
dc = DoeclimCalib(name='doeclim_sim', mode=MODE_SIM, vin=vin)

xs = [[3,10,0.5]+[0]*7,
      [3,10,0.5]+[1]*7]

a = dc.run(xs[0], time=Time(1900, 1949))
b = dc.run(xs[0], time=Time(1950, 2015), eval_eqs_notime=False)
c = dc.run(xs[0])

#xs = itertools.product([1, 9], [0.01, 100], [-1, 1])
fig, axs = plt.subplots(2,4)
for a in [b,c]:
    for v,ax in zip(['DQ1', 'DQ2', 'DelQL', 'DelQO', 'DPAST2', 'DTEAUX1', 'DTEAUX2'], axs.flat):
        getattr(a, v).plot(ax=ax)
        ax.set_title(v)

