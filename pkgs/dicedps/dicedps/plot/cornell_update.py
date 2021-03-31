from dicedps.plot.common import *

myload = lambda x, *args: Data.load(inrootdir('dicedps','data',x), *args)
inplot = lambda x: os.path.join(os.environ['HOME'],'working','presentation-egu2018-dice-dps','figures', x)

plotdir = lambda x: os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations', '2017-11-Update-Cornell', 'figures', x)

dice = lambda: Dice(time=Time(start=2015, periods=50, tstep=5))

bau = myload('bau', lambda: dice().set_bau().solve())
opt = myload('opt', lambda: dice().solve())
opt_wrong = myload('opt_wrong', lambda: dice().fix('MIU',opt).fix('S',opt).set(t2xco2=4.7).solve())
opt_hadweknown = myload('opt_hadweknown', lambda: dice().set(t2xco2=4.7).solve())

from pyomo.core.base import Expression
def fix2ubound(m,t):
    if m.isfixed('MIU', t): return Expression.Skip
    m.setfixed('MIU', t)
    return max(opt.MIU.loc[t], min(1,0.2*(t-1)))
    #return min(1.2, 0.2*(t-1)) #if m.year[t] < 2050 else 1.2
def fix2lbound(m,t):
    if m.isfixed('MIU', t): return Expression.Skip
    m.setfixed('MIU', t)
    return 0.
greenpeace = myload('greenpeace', lambda: dice().add_rule('MIU', fix2ubound).solve())
trump = myload('trump', lambda: dice().add_rule('MIU', fix2lbound).solve())

#plt.style.use(['seaborn-paper','seaborn-darkgrid'])
coeff = {'MIU':100.}
def plot_dice_panel(axs, y, ylab, **kws):
    for ax, v, vlab, ylim in zip(axs.flat,
                                 ['MIU', 'C_DIFFPCT_BAU', 'TATM'],
                                 ['Abatement (%)', 'Consumption loss wrt No CC (%)', 'Temperature increase (K)'],
                                 [[-10, 130], [-1, 20], [0, 8]]):
        (coeff.get(v, 1.)*getattr(y, v + '_year')).loc[:2200].plot(ax=ax, label=ylab, lw=2, **(kws.get(v, {})))
        ax.set_title(vlab)
        ax.set_ylim(ylim)
        ax.legend()
        ax.set_xlabel('Year')

for i in range(1,4):
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    for y, ylab in zip([opt, greenpeace, trump][:i],
                       ['Nordhaus', 'Paris-agreement inspired', 'Club for Growth inspired']):
        plot_dice_panel(axs, y, ylab)
    fig.tight_layout()
    sb.despine(fig)
    fig.savefig(inplot(f'policy_alt{i}.pdf'))


for i in range(1,4):
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    for y, ylab in zip([opt, opt_wrong, opt_hadweknown][:i],
                       ['Base Optimal | Mean ECS', 'Base Optimal | Worse ECS', 'Optimal | Worse ECS']):
        if ylab == 'Base Optimal | Worse ECS':
            kws = {'MIU':{'ls':'--'}}
        else:
            kws = {}
        plot_dice_panel(axs, y, ylab, **kws)
    fig.tight_layout()
    sb.despine(fig)
    fig.savefig(inplot(f'policy_opt{i}.pdf'))

opt_hadweknown.CEMUTOTPER.sum()
opt_wrong.CEMUTOTPER.sum()
opt.CEMUTOTPER.sum()
import pandas as pd
pd.concat({'bau':bau.C,'opt':opt.C},1).plot()