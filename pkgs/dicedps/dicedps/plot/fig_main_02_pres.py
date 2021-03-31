import string
from typing import Dict

from dicedps.plot.common import *

df = load_merged(orient='min')

clp()
sb.set_context('notebook', font_scale=1.1)


# Simulators
simdps = get_sim2plot(mdps, 100)
simtime = get_sim2plot(mtime, 100)
simdps10k = get_sim2plot(mdps, 1000)
simtime10k = get_sim2plot(mtime, 1000)


xmin, xmax = 30, 360
# Diff

dftime = df.loc[mtime]
dfdps = df.loc[mdps]
dfdiff = {}
for i, (k, ndec) in enumerate(zip([o.o_min_cbgemitcost_lab, o.o_min_mean2degyears_lab], [2,0])):
    ok = [o.o_min_mean2degyears_lab,o.o_min_cbgemitcost_lab][i]
    osort = [k]+[ok]
    dftime_thinned = dftime.sort_values(osort).round({k:ndec}).groupby(k).first()
    dfdps_thinned = dfdps.sort_values(osort).round({k:ndec}).groupby(k).first()
    dfdiff[k] = pd.concat([dftime_thinned[ok],dfdps_thinned[ok]], keys=[mtime,mdps], axis=0).swaplevel(0,1).sortlevel()


"""
df.groupby([o.o_min_cbgemitcost_lab].head()
dfdiff.head()
x = np.round(np.arange(xmin,xmax,0.1),1)
ret = {}
for xx in x:
    y = get_sol(df, {o.o_min_mean2degyears_lab:xx, o.o_min_cbgemitcost_lab:'min'}).unstack(1)
    if v.get_o(y).dropna(1).shape[1] == 0:
        continue
    ret[xx] = y
dfdiff = pd.concat(ret)
dfdiff.head()
highlev = 34.
y = dfdiff.loc[highlev]
A = y.loc[mdps]
"""

df09mit = df[np.isclose(df[o.o_min_cbgemitcost_lab],0.9,atol=1e-4)]

hpoints = {}
hpoints['A'] = df09mit.loc[mdps].sort_values(o.o_min_mean2degyears_lab).iloc[0]
hpoints['N1'] = df09mit.loc[mtime].sort_values(o.o_min_mean2degyears_lab).iloc[0]
hpoints['N2'] = df[np.isclose(df[o.o_min_mean2degyears_lab], hpoints['A'][o.o_min_mean2degyears_lab], atol=2e-2)].loc[mtime].sort_values(o.o_min_cbgemitcost_lab).iloc[0]

write_macro = lambda *x, **kws: x

AN1_mitcosts_pct = [hpoints[x][o.o_min_cbgemitcost_lab] for x in ['A', 'N1']]
AN1_mean_mitcost_pct = sum(AN1_mitcosts_pct) / 2.
print(f"fig02-ex-mitcost-pct: {AN1_mitcosts_pct}")
write_macro('fig02-ex-mitcost-pct', f"{AN1_mean_mitcost_pct:.1f}", reset=True)
abau = simtime.dc.run(np.zeros(47))
cons_2015_tusd = abau.C.loc[2015].mean()
AN1_mean_mitcost_tusd = cons_2015_tusd*AN1_mean_mitcost_pct/100.
write_macro('fig02-ex-mitcost-Tusd', f"{AN1_mean_mitcost_tusd:.2f}")
N1_2dy = hpoints['N1'][o.o_min_mean2degyears_lab]
A_2dy = hpoints['A'][o.o_min_mean2degyears_lab]
N1mA_2dy = -(A_2dy-N1_2dy)
N1mA_2dy_pct = N1mA_2dy/N1_2dy*100.
write_macro('fig02-N1-2degyears', f'{np.round(N1_2dy, -1):.0f}')
write_macro('fig02-N1-minus-A-2degyears', f'{N1mA_2dy:.0f}')
write_macro('fig02-N1-minus-A-2degyears-pct', f'{N1mA_2dy_pct:.0f}')
AN2_2dys = [hpoints[x][o.o_min_mean2degyears_lab] for x in ['A', 'N2']]
print(f'test: {AN2_2dys}')
AN2_mean_2dy = sum(AN2_2dys)/2.
write_macro('fig02-ex-2degyears', f'{np.round(AN2_mean_2dy, -1):.0f}')
N2_mitcost_pct = hpoints['N2'][o.o_min_cbgemitcost_lab]
A_mitcost_pct = hpoints['A'][o.o_min_cbgemitcost_lab]
write_macro('fig02-N2-mitcost-pct', f"{N2_mitcost_pct:.1f}")
write_macro('fig02-A-mitcost-pct', f"{A_mitcost_pct:.1f}")
N2mA_mitcost_Tusd = cons_2015_tusd*(N2_mitcost_pct-A_mitcost_pct)/100
write_macro('fig02-N2-minus-A-mitcost-Busd', f"{(1000*np.round(N2mA_mitcost_Tusd,2)):.0f}")
clp()

plot2plot = 1  # 0 = temp, 1 = mitcost

clp()
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(4, 7)
ax_pareto = plt.subplot(gs[:3, :3])
ax_diff = plt.subplot(gs[3,:3])
ax_diff2 = plt.subplot(gs[:3,3])
ax_temp = plt.subplot(gs[:,4:])

prop_list = list(prop_cycle())
prop_list[0]['color'] = prop_list[1]['color']
prop_list[1]['color'] = '0.5'
getxy = lambda s: hpoints[s][[o.o_min_mean2degyears_lab, o.o_min_cbgemitcost_lab]].values
# Pairs
plot_objective_pairs(df, orows=[o.o_min_cbgemitcost_lab], ocols=[o.o_min_mean2degyears_lab], axs=ax_pareto, prop_list=prop_list)
ax_pareto.set_xscale('log')
ax_pareto.set_xticks([30,50,100,200,300])
ax_pareto.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax_pareto.set_xlim([25,350])


titleg = 'From panel (a)'
ydiff = dfdiff[o.o_min_mean2degyears_lab].unstack(1).diff(axis=1)[mtime].dropna().mul(cons_2015_tusd*1000/100)
ax_diff.plot(ydiff.index, ydiff.values, lw=2, **prop_list[0])
#ydiff.plot(ax=ax_diff, legend=False, prop)
ax_diff.set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
ax_diff.set_yscale('linear')
ax_diff.set_xscale('log')
ax_diff.set_xticks([30,50,100,200,300])
ax_diff.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax_diff.set_xlim([25,350])
ax_diff.set_ylabel(f'VOI (Billion $)')
ax_diff.set_ylim([0,150])


ydiff = dfdiff[o.o_min_cbgemitcost_lab].unstack(1).diff(axis=1)[mtime].dropna()
ydiffsort = ydiff.reset_index().sort_values(o.o_min_cbgemitcost_lab)

ydiffsmooth = ydiffsort.set_index(o.o_min_cbgemitcost_lab).rolling(5, min_periods=1).mean()['time']
ax_diff2.plot(ydiffsmooth.values, ydiffsmooth.index.values, lw=2, **prop_list[0])
#ax_diff.scatter(dfdiff.index.levels[0],
#                1000./100.*cons_2015_tusd*((dfdiff.xs(mtime,0,1)[o.o_min_cbgemitcost_lab].values)-
#                (dfdiff.xs(mdps,0,1)[o.o_min_cbgemitcost_lab].values)), s=5, color='k')
ax_diff2.set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
ax_diff2.set_yscale('linear')
ax_diff2.set_ylim(ax_pareto.get_ylim())
ax_diff2.set_xscale('linear')
#ax_diff2.set_xticks([30,50,100,200,300])
ax_diff2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax_diff2.set_xlim([0.1,10])
ax_diff2.set_xlabel(f'VOI (K-year)')
#ax_diff2.set_ylim([0,150])


if plot2plot == 0:
    xa,ya = getxy('A')
    hhoriz_par = ax_pareto.axhline(ya, color='k', ls='--')
    hhoriz_diff = ax_diff2.axhline(ya, color='k', ls='--')
    ha_annot = ax_pareto.annotate('A', xy=(xa, ya), va='top', ha='right')
    ha_point = ax_pareto.scatter(xa, ya, s=10, color='k')
    xa,ya = getxy('N1')
    hn1_annot = ax_pareto.annotate('N', xy=(xa, ya), va='bottom', ha='left')
    hn1_point = ax_pareto.scatter(xa, ya, s=10, color='k')
else:
    xa,ya = getxy('A')
    hhoriz_par = ax_pareto.axvline(xa, color='k', ls='--')
    hhoriz_diff = ax_diff.axvline(xa, color='k', ls='--')
    ha_annot = ax_pareto.annotate('A', xy=(xa, ya), va='top', ha='right')
    ha_point = ax_pareto.scatter(xa, ya, s=10, color='k')
    xa,ya = getxy('N2')
    hn2_annot = ax_pareto.annotate('N', xy=(xa, ya), va='bottom', ha='left')
    hn2_point = ax_pareto.scatter(xa, ya, s=10, color='k')


if plot2plot == 0:
    temp2100: Dict[str, pd.DataFrame] = {}
    for s, lab, sim, pro in zip(['N1', 'A'], ['Non-adaptive', 'Adaptive'], [simtime10k, simdps10k], [prop_list[1],prop_list[0]]):
        p = hpoints[s]
        sim.dc.run_and_ret_objs(v.get_x(p))
        temp2100[s] = sim.get('TATM').loc[2100]
        sb.distplot(temp2100[s], ax=ax_temp, label=lab, kde=True, kde_kws={"shade": True}, hist=False, **pro) # kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)

    hleg, lleg = ax_temp.get_legend_handles_labels()
    ax_temp.legend(list(reversed(hleg)), list(reversed(lleg)))

    ax_temp.set_xlabel(lab_temp_year(2100))
    ax_temp.set_ylabel('PDF')
else:
    ax_mitcost = ax_temp
    for s, lab, sim, pro in zip(['N2', 'A'], ['Non-adaptive', 'Adaptive'], [simtime10k, simdps10k], [prop_list[1],prop_list[0]]):
        p = hpoints[s]
        sim.dc.run(v.get_x(p))
        if s == 'N2':
            ax_mitcost.axvline(p[o.o_min_cbgemitcost_lab], **pro)
        else:
            h = sb.distplot(Dice.cbge_mitcost_v1(sim.dc._mlist[1]), ax=ax_mitcost, label=s, kde=True, kde_kws={"shade": True}, hist=False, **pro)  #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)
    handles, labels = ax_mitcost.get_legend_handles_labels()
    handles = [handles[0]] + [plt.Line2D([0], [0], lw=1.5, **prop_list[1])]
    labels =  ['Adaptive', 'Non-adaptive']
    ax_mitcost.legend(handles, labels)
    ax_mitcost.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab].replace('\n',' '))
    ax_mitcost.set_ylabel('PDF')



sb.despine(fig)

fig.tight_layout()

ax_temp.set_visible(True)

if plot2plot == 0:
    fig.savefig('fig-pareto-2d-full-temp.pdf')
else:
    fig.savefig('fig-pareto-2d-full-mitcost.pdf')


ax_temp.set_visible(False)
if plot2plot == 0:
    fig.savefig('fig-pareto-2d-50-temp.pdf')
else:
    fig.savefig('fig-pareto-2d-50-mitcost.pdf')


for hh in [hhoriz_par, hhoriz_diff, ha_annot, ha_point, hn1_annot, hn1_point]:
    hh.set_visible(False)
fig.savefig('fig-pareto-2d-40-temp.pdf')

for hh in [ax_diff, ax_diff2]:
    hh.set_visible(False)
fig.savefig('fig-pareto-2d-30-temp.pdf')







ax_diff2.set_visible(False)
ax_diff.set_visible(False)

fig.savefig('fig-pareto-2d-pair.pdf')

#ydiff = dfdiff.loc[highlev][o.o_min_cbgemitcost_lab]
#ax_diff.annotate('N - A', xy=(highlev, ydiff[mtime]-ydiff[mdps]), va='bottom', ha='left')
#ax_diff.scatter([highlev], [ydiff[mtime]-ydiff[mdps]], s=10, **prop_list[2])
#ax_diff.set_xlim([xmin,xmax])

dfdiff[o.o_min_cbgemitcost_lab].unstack(1).head()
0.14*cons_2015_tusd
plot_var_cmap(simdps, v.get_x(hpoints['A']), yy=['MIU'], axs=ax_cmap, pad='7%')
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
ax_cmap.legend(hlist, llist, title=titleg)

temp2100: Dict[str, pd.DataFrame] = {}
for s, sim, pro in zip(['A','N1'], [simdps10k, simtime10k], prop_list):
    p = hpoints[s]
    sim.dc.run_and_ret_objs(v.get_x(p))
    temp2100[s] = sim.get('TATM').loc[2100]
    sb.distplot(temp2100[s], ax=ax_temp, label=s, kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)


ax_temp.legend(title=titleg)
ax_temp.set_xlabel(lab_temp_year(2100))
ax_temp.set_ylabel('PDF')


for s, sim, pro in zip(['A', 'N2'], [simdps10k, simtime10k], prop_list):
    p = hpoints[s]
    sim.dc.run(v.get_x(p))
    if s == 'N2':
        ax_mitcost.axvline(p[o.o_min_cbgemitcost_lab], **pro)
    else:
        h = sb.distplot(Dice.cbge_mitcost_v1(sim.dc._mlist[1]), ax=ax_mitcost, label=s, kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)
handles, labels = ax_mitcost.get_legend_handles_labels()
handles = [handles[0]] + [Line2D([0], [0], lw=1.5, **prop_list[1])]
labels = ['A','N2']
ax_mitcost.legend(handles, labels, title=titleg)
ax_mitcost.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax_mitcost.set_ylabel('PDF')

hletters = []
for i, ax in enumerate([ax_pareto, ax_cmap, ax_temp, ax_mitcost]):
    sb.despine(fig, ax)
    hletters.append(ax.text(0.95, 1., string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))
fig.tight_layout()

savefig4paper(fig, 'main_02')
