import string
from collections import defaultdict

from dicedps.plot.common import *
from paradice.dice import Damages

simtime = get_sim2plot(mtime, cli='high', obj_set='v3')

import dicedps.qsub_dice as q
q.name2dargs('u1w1000doeclim_mrbfXdX41_i1p200_nfe4000000_objv2_cnone_s0_seed0000_merged.csv')

simtime = get_sim2plot(mtime, cli='high', obj_set='v3')
#simdps = get_sim2plot(mdps)

simtime.dc.run_and_ret_objs(np.zeros(47))



m = simtime.dc._mlist[1]
damcosts = pd.Series(-100*(pow(Dice.welfare_damcost(m)/Dice.welfare_ref(m), 1/(1 - m.elasmu)) - 1))
#.quantile(0.95) #.describe(percentiles=[0.95])

sb.distplot(damcosts)
plt.axvline(Dice.cbge_damcost(m))
plt.axvline(damcosts.mean(), color='k')
Dice.bge_damcost_q95(m)

dfnom.index.levels[0]

y = {}
for s, m in zip([simdps, simtime], [mdps, mtime]):
    dfnomx = dfnom.loc[m]
    dfnomxs = dfnomx.sort_values([o.o_min_cbgemitcost_lab, o.o_min_mean2degyears_lab])
    y[m] = dfnomxs.groupby(np.round(dfnomxs[o.o_min_cbgemitcost_lab], 2)).first()

for a, b in y.iterrows():
    simdps.dc.run(v.get_x(b))
    break

ytemp = {}
ytempdiff = {}
for s, m in zip([simdps, simtime], [mdps, mtime]):
    s.dc.run(v.get_x(y[m].loc[1]))
    ytemp[m] = s.get('TATM')
    ytempdiff[m] = ytemp[m].diff()

fig, ax = plt.subplots(1, 1, figsize=(w2col,hhalf))
for m, p in zip([mtime, mdps], prop_list):
    for isow in ytemp[m].columns:
        ax.plot(ytemp[m][isow], ytempdiff[m][isow], **p)


fig, axs = plt.subplots(1, 2, figsize=(w2col,hhalf))
for m, ax in zip([mtime, mdps], axs):
    sb.scatterplot(x=o.o_min_miu2030_lab, y=o.o_min_miu2050_lab, hue=o.o_min_mean2degyears_lab, data=y[m], ax=ax)
