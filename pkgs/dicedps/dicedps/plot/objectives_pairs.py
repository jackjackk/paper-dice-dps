from dicedps.plot.common import *

inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','paper-dice-dps','meetings','20180607-keller-group-update','figures',*x)

df = load_merged(orient='min')

clp()

# Simulators
simdps = get_sim2plot(mdps, 100)
simtime = get_sim2plot(mtime, 100)
simdps1k = get_sim2plot(mdps, 1000)
simtime1k = get_sim2plot(mtime, 1000)

"""
xmin, xmax = 21.240221450112973, 373.49247252276695
# Diff
x = np.round(np.arange(xmin,xmax),0)
ret = {}
for xx in x:
    y = get_sol(df, {o.o_min_mean2degyears_lab:xx, o.o_min_cbgemitcost_lab:'min'}).unstack(1)
    if v.get_o(y).dropna(1).shape[1] == 0:
        continue
    ret[xx] = y
dfdiff = pd.concat(ret)

highlev = 35.8
y = dfdiff.loc[highlev]
A = y.loc[mdps]
"""
df09mit = df[np.isclose(df[o.o_min_cbgemitcost_lab],0.9,atol=1e-4)]

hpoints = {}
hpoints['A'] = df09mit.loc[mdps].sort_values(o.o_min_mean2degyears_lab).iloc[0]
hpoints['N1'] = df09mit.loc[mtime].sort_values(o.o_min_mean2degyears_lab).iloc[0]
hpoints['N2'] = df[np.isclose(df[o.o_min_mean2degyears_lab], hpoints['A'][o.o_min_mean2degyears_lab], atol=2e-2)].loc[mtime].sort_values(o.o_min_cbgemitcost_lab).iloc[0]

clp()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12,7))
gs = GridSpec(4, 3)
ax_pareto = plt.subplot(gs[:2, :2])
#ax_diff = plt.subplot(gs[2, :])
ax_cmap = plt.subplot(gs[2:,:2])
ax_temp = plt.subplot(gs[:2,2])
ax_mitcost = plt.subplot(gs[2:,2])

getxy = lambda s: hpoints[s][[o.o_min_mean2degyears_lab, o.o_min_cbgemitcost_lab]].values
# Pairs
plot_objective_pairs(df, orows=[o.o_min_cbgemitcost_lab], ocols=[o.o_min_mean2degyears_lab], axs=ax_pareto)
xa,ya = getxy('A')
for s, va, ha in zip(['N1','N2','A'], ['bottom','bottom','top'],['left','left','right']):
    xhigh, yhigh = getxy(s)  #y.loc[miu, o.o_min_mean2degyears_lab], y.loc[miu, o.o_min_cbgemitcost_lab]
    if s!='A':
        ax_pareto.plot([xa,xhigh],[ya,yhigh], ls='--',color='k')
    ax_pareto.annotate(s, xy=(xhigh, yhigh), va=va, ha=ha)
    ax_pareto.scatter(xhigh, yhigh, s=10, color='k')

#ax_pareto.axvline(highlev, ls='--', color='.5')

"""
ax_diff.scatter(dfdiff.index.levels[0],dfdiff.xs(mtime,0,1)[o.o_min_cbgemitcost_lab].values-dfdiff.xs(mdps,0,1)[o.o_min_cbgemitcost_lab].values, s=5, color='k')
ax_diff.set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
ax_diff.set_ylabel(f'Value of\ninformation')
ydiff = dfdiff.loc[highlev][o.o_min_cbgemitcost_lab]
ax_diff.annotate('N - A', xy=(highlev, ydiff[mtime]-ydiff[mdps]), va='bottom', ha='left')
ax_diff.scatter([highlev], [ydiff[mtime]-ydiff[mdps]], s=10, **prop_list[2])
ax_diff.set_xlim([xmin,xmax])
"""

plot_var_cmap(simdps, y, yy=['MIU'], axs=ax_cmap)
ax_cmap.set_xlabel('Year')
hlist = []
from matplotlib.lines import Line2D
cmap = simdps.cmap
hlist = [Line2D([0], [0], color=cmap(1.), lw=1.5),
         Line2D([0], [0], color=cmap(0.), lw=1.5)]
llist = ['A, high climate sensitivity', 'A, low climate sensitivity']
for s, ls in zip(['N1','N2'], [':','--']):
    simtime.dc.run(v.get_x(hpoints[s]))
    hlist.append(ax_cmap.plot(simtime.get('MIU'), lw=1.5, ls=ls, color='k')[0])
    llist.append(s)
ax_cmap.legend(hlist, llist)

for s, sim, pro in zip(['A','N1'], [simdps1k, simtime1k], prop_list):
    p = hpoints[s]
    sim.dc.run_and_ret_objs(v.get_x(p))
    sb.distplot(sim.get('TATM').loc[2100], ax=ax_temp, label=s, **pro)
ax_temp.legend()
ax_temp.set_xlabel('Temperature in 2100 (K)')
ax_temp.set_ylabel('PDF')

for s, sim, pro in zip(['A', 'N2'], [simdps1k, simtime1k], prop_list):
    p = hpoints[s]
    sim.dc.run(v.get_x(p))
    if s == 'N2':
        ax_mitcost.axvline(p[o.o_min_cbgemitcost_lab], **pro)
    else:
        h = sb.distplot(Dice.cbge_mitcost_v1(sim.dc._mlist[1]), ax=ax_mitcost, label=s, **pro)
handles, labels = ax_mitcost.get_legend_handles_labels()
handles = [handles[0]] + [Line2D([0], [0], lw=1.5, **prop_list[1])]
labels = ['A','N2']
ax_mitcost.legend(handles, labels)
ax_mitcost.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax_mitcost.set_ylabel('PDF')

fig.tight_layout()
fig.savefig(inplot('figure02.png'), dpi=250)
    #ax_mitcost.set_ylim([0,2])



for ax in axs_cmap2:
    ax.set_ylabel('')
for ax in [axs_cmap1[0], axs_cmap2[0]]:
    ax.set_xlabel('')
for ax in [axs_cmap1[1], axs_cmap2[1]]:

fig.tight_layout()

"""
dc = dice_last(mtime, 1000)

dc.run_and_ret_objs(np.zeros(47))

dc.TATM.iloc[-11].describe(percentiles=[.95])

objlist = o.oset2labs['greg4d']
plot_objective_pairs(df, orows=objlist, ocols=objlist)

fig, ax = plt.subplots(1,1,figsize=(12,6))
plot_objective_pairs(df, orows=[o.o_min_cbgemitcost_lab], ocols=[o.o_min_mean2degyears_lab], axs=ax)
clp()

v.get_ocols(df)
get_sol_by_mitcost()
"""