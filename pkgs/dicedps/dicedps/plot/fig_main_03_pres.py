import string
from collections import defaultdict

from dicedps.plot.common import *
from paradice.dice import Damages

df = load_rerun('greg4h', orient='min')

simtime = get_sim2plot(mtime)
simdps = get_sim2plot(mdps)

#inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','paper-dice-dps','meetings','20180607-keller-group-update','figures',*x)


dfnom = df.loc['med1']

dfnom.index.levels[0]

y = {}
for s, m in zip([simdps, simtime], [mdps, mtime]):
    dfnomx = dfnom.loc[m]
    dfnomxs = dfnomx.sort_values([o.o_min_cbgemitcost_lab, o.o_min_mean2degyears_lab])
    y[m] = dfnomxs.groupby(np.round(dfnomxs[o.o_min_cbgemitcost_lab], 2)).first()

#for a, b in y.iterrows():
#    simdps.dc.run(v.get_x(b))
#    break

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

write_macro = lambda *x, **kws: x

m = simtime.dc._mlist[1]
m.TATM[0,:] = np.linspace(1,5,100)

prop_list2 = [prop_list[i] for i in [2,4,3]]+[{'color':'0.5'}]

#mpl.rcParams.update(mpl_nature)
clp()


sb.set_context('notebook', font_scale=1.1)

plot2plot = 0
fig = plt.figure(figsize=(12,6))

gs = GridSpec(2, 3)
ax_damfunc = plt.subplot(gs[0, 2])
ax_cs = plt.subplot(gs[1,2])
ax_troff = plt.subplot(gs[:,:2])

ldam = ['Nordhaus', 'Weitzman', 'Goes']
ldam2 = ldam  #['Low', 'Medium', 'High']


for i in range(3):
    ax_damfunc.plot(m.TATM[0], 100*Damages.dam2func[i+1](m, 0), label=ldam2[i], **prop_list2[i])
hleg = ax_damfunc.legend([],[], title='Damage Function (DF)')
plt.setp(hleg.get_title(),fontsize=10)
ax_damfunc.set_xlabel('Temperature (K)')
ax_damfunc.set_ylabel('GDP loss (%)')



for cslev, cslab, p in zip(['low','med','hig'], ['Low', 'Mid', 'High'], prop_list2):
    dfcs = u.get_sows_setup_mcmc(h.args2climcalib(cslev), nsow=1000)
    sb.distplot(dfcs['setup']['t2co'], label=cslab, ax=ax_cs, kde=True, kde_kws={"shade": True, 'lw': 2}, hist=False, **p) #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **p)
hleg = ax_cs.legend([], [], title='Climate\nSensitivity (CS)')
plt.setp(hleg.get_title(),fontsize='small')
ax_cs.set_xlabel('Climate sensitivity (K)')
ax_cs.set_ylabel('PDF')


xcol = o.o_min_cbgemitcost_lab
ycol = o.o_min_cbgedamcost_lab
ycol = o.o_min_q95damcost_lab
scol = o.o_min_mean2degyears_lab
xcol_ndec = 3
ids2plot = {}
for miu in [mdps,mtime]:
    dfnom = df.loc['med1'].loc[miu]
    dfnom_sorted = dfnom.sort_values(scol).reset_index()
    ids2plot[miu] = dfnom_sorted.groupby(np.round(dfnom_sorted[xcol],xcol_ndec)).first()['idsol']

scen2lab = {
    'low1': 'Low CS + Low DF',
    'med2': 'Mid CS + Mid DF',
    'hig3': 'High CS + High DF',
    'med1': 'Nominal',
}

stidy = lambda x: x #lambda x: x.replace('CS', 'climate sensitivity').replace('DF', 'damage function').replace(' + ', '\n')

scen2off = {
    'med1': 5,
}
hs = []
ydict = defaultdict(dict)
yotherdict = defaultdict(dict)
scenlist = ['low1','med2','hig3','med1']
hsannot = []
hsdiff = []
hsdiffannot = []
for scen, p in zip(scenlist, prop_list2):
    k = o.o_min_cbgemitcost_lab
    ok = o.o_min_q95damcost_lab
    osort = [k] + [ok]
    dftime_thinned = df.loc[scen].loc[mtime].sort_values(osort).round({k: 2}).groupby(k).first()
    dfdps_thinned = df.loc[scen].loc[mdps].sort_values(osort).round({k: 2}).groupby(k).first()
    dfdiff = pd.concat([dftime_thinned[ok], dfdps_thinned[ok]], keys=[mtime, mdps], axis=0).swaplevel(0, 1).sortlevel()
    for miu, ls in zip([mdps, mtime], ['-','--']):
        dfcurr = df.loc[scen].loc[miu].loc[ids2plot[miu]]
        #ax_troff.scatter(dfcurr[xcol], dfcurr[ycol], s=3, **p)
        y2plot = pd.Series(dfcurr[ycol].values,index=dfcurr[xcol].round(2)).groupby(o.o_min_cbgemitcost_lab).mean().reindex(np.arange(0,1.25,1e-2)).interpolate()
        yother = pd.Series(dfcurr[xcol].values, index=dfcurr[ycol].round(2)).groupby(
            ycol).mean().reindex(np.arange(0, 23, 1e-2)).interpolate()
        ydict[scen][miu]=y2plot
        yotherdict[scen][miu]=yother
        #sb.regplot(dfcurr[xcol], dfcurr[ycol], order=8, line_kws=dict(lw=1.5, ls=ls), scatter_kws=dict(s=3), **p)
        #hs.append(ax_troff.plot(dfcurr[xcol], dfcurr[ycol], ls=ls, lw=1.5, **p)[0])
        hs.append(ax_troff.plot(y2plot.index, y2plot.values, ls=ls, lw=2, **p)[0])
    hsannot.append(ax_troff.annotate(s=stidy(scen2lab[scen]), xy=(dfcurr[xcol].iloc[-1], dfcurr[ycol].iloc[-1]), xytext=(0,5), textcoords='offset points', ha='right', va='bottom', fontsize=10))
    ydiff = dfdiff.unstack(1).reindex(np.arange(0,1.21,0.01)).interpolate().diff(axis=1)['time'].rolling(3, min_periods=1).mean()
    hsdiff.append(ax_troff.plot(ydiff.index, np.maximum(0.02,ydiff.values), lw=2, **p)[0])
    hsdiffannot.append(ax_troff.annotate(s=stidy(scen2lab[scen]), xy=(ydiff.index[-1], max(0.02, ydiff.values[-1])), xytext=(0,5), textcoords='offset points', ha='right', va='bottom', fontsize=10))
ax_troff.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax_troff.set_ylabel(obj2lab2[o.o_min_q95damcost_lab].replace(' (',' \n('))
ax_troff.set_yscale('log')
#ax_troff.locator_params(nbins=2, axis='y')
ax_troff.set_yticks([0.1, 0.2,0.4,0.8,1,2,4,8,10,20,40])
ax_troff.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax_troff.set_ylim([0.02, 50])
#ax_troff.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
#ax_troff.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
#ax_troff.ticklabel_format(style='plain',axis='y',useOffset=False)
#ax_troff.get_yaxis().get_major_formatter().labelOnlyBase = False


for scen, p in zip(scenlist, prop_list2):
    break

fig.tight_layout()

sb.despine(fig)

for hh in hs+hsannot:
    hh.set_visible(False)

hleg_troff = ax_troff.legend([], [], title='Value of Information')
plt.setp(hleg_troff.get_title(),fontsize=10)

fig.savefig(inplot('fig-dam95th-70-voi.pdf'))

for hh in hs+hsannot:
    hh.set_visible(True)

ax_troff.legend(hs[-2:],[miu2lab[mdps],miu2lab[mtime]])

for hh in hsdiff+hsdiffannot:
    hh.set_visible(False)
hsdiff
fig.savefig(inplot('fig-dam95th-50-full.pdf'))

hl, ll = ax_damfunc.get_legend_handles_labels()
for ih, hh in enumerate(hl):
    hh.set_visible(ih == 0)

hleg, ll = ax_cs.get_legend_handles_labels()
for hl in hleg, ax_cs.collections:
    for ih, hh in enumerate(hl):
        hh.set_visible(ih == 1)

for ih, hh in enumerate(hs[:-2]):
    hh.set_visible(False)

for hh in hsannot:
    hh.set_visible(False)

fig.savefig(inplot('fig-dam95th-30-nom.pdf'))

hletters = []
for i, ax in enumerate([ax_damfunc, ax_cs, ax_troff]):
    sb.despine(fig, ax)
    hletters.append(ax.text(0.95, 1., string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))
fig.tight_layout()

savefig4paper(fig, 'main_03')

min_95damcost_pct = df.loc[scenlist][ycol].min()
max_95damcost_pct = df.loc[scenlist][ycol].max()
abau = simtime.dc.run(np.zeros(47))
cons_2015_tusd = abau.C.loc[2015].mean()
min_95damcost_tusd = cons_2015_tusd*min_95damcost_pct/100
max_95damcost_tusd = cons_2015_tusd*max_95damcost_pct/100
write_macro('fig03-min-95damcost-pct', f'{min_95damcost_pct:.1f}', True)
write_macro('fig03-min-95damcost-Tusd', f'{min_95damcost_tusd:.1f}')
write_macro('fig03-max-95damcost-pct', f'{max_95damcost_pct:.1f}')
write_macro('fig03-max-95damcost-Tusd', f'{max_95damcost_tusd:.1f}')

"""
a.loc[a.argmax()]
a.describe()
a.reset_index()
sb.factorplot(x='mitcost',y='time',hue='scen',order=np.arange(0,23,1e-1),data=a.reset_index())
"""

#### SUPP PLOT

fig = plt.figure(figsize=(w2col,1.5*hhalf))
gs = GridSpec(4, 2)
ax_troff = plt.subplot(gs[:2, :])  # a. Trade-off 95th dam cost vs mit cost
ax_x1 = [plt.subplot(gs[x,0]) for x in [2,3]]  # b-c. Abatement and damage
ax_x2 = [plt.subplot(gs[x,1]) for x in [2,3]]  # d-e.

hs = []
ydict = defaultdict(dict)
yotherdict = defaultdict(dict)
scenlist = ['low1','med2','hig3','med1']
for scen, p in zip(scenlist, prop_list2):
    for miu, ls in zip([mdps, mtime], ['-','--']):
        dfcurr = df.loc[scen].loc[miu].loc[ids2plot[miu]]
        #ax_troff.scatter(dfcurr[xcol], dfcurr[ycol], s=3, **p)
        y2plot = pd.Series(dfcurr[ycol].values,index=dfcurr[xcol].round(2)).groupby(o.o_min_cbgemitcost_lab).mean().reindex(np.arange(0,1.25,1e-2)).interpolate()
        yother = pd.Series(dfcurr[xcol].values, index=dfcurr[ycol].round(2)).groupby(
            ycol).mean().reindex(np.arange(0, 23, 1e-2)).interpolate()
        ydict[scen][miu]=y2plot
        yotherdict[scen][miu]=yother
        #sb.regplot(dfcurr[xcol], dfcurr[ycol], order=8, line_kws=dict(lw=1.5, ls=ls), scatter_kws=dict(s=3), **p)
        #hs.append(ax_troff.plot(dfcurr[xcol], dfcurr[ycol], ls=ls, lw=1.5, **p)[0])
        hs.append(ax_troff.plot(y2plot.index, y2plot.values, ls=ls, lw=1.5, **p)[0])
    ax_troff.annotate(s=stidy(scen2lab[scen]), xy=(dfcurr[xcol].iloc[-1], dfcurr[ycol].iloc[-1]), xytext=(0,scen2off.get(scen, 15)), textcoords='offset points', ha='right', va='bottom')
ax_troff.legend(hs[-2:],[miu2lab[mdps],miu2lab[mtime]])
ax_troff.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax_troff.set_ylabel(obj2lab2[o.o_min_q95damcost_lab].replace('(','\n('))
ax_troff.set_yscale('log')
ax_troff.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
ax_troff.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.1f'))


a=pd.concat({s:pd.concat(yotherdict[s]) for s in scenlist},names=['scen','miu','mitcost']).unstack('miu').diff(axis=1).iloc[:,1]
amax = a.argmax()
b=df.loc[amax[0]].loc[mdps]
bdamcostmin=b[o.o_min_q95damcost_lab].min()

simdps = get_sim2plot(mdps, 100, cli=amax[0][:3], damfunc=amax[0][-1])
flatpolicies_by_mitcost = b[np.isclose(b[o.o_min_q95damcost_lab], bdamcostmin) & (b[o.o_min_cbgemitcost_lab]<=1.25)].sort_values(o.o_min_cbgemitcost_lab)
for pol_idx, axs, lab in zip([0,-1], [ax_x1, ax_x2], ['A1', 'A2']):
    pol = flatpolicies_by_mitcost.iloc[pol_idx]
    plot_var_cmap(simdps,
                  amiu=v.get_x(pol),
                  yy=['MIU','TATM'],
                  axs=axs)
    ax_troff.scatter(pol[xcol], pol[ycol], color='k')
    ax_troff.annotate(lab, xy=(pol[xcol], pol[ycol]), xytext=(5, 5), textcoords='offset pixels', va='bottom', ha='left', fontsize=8)
    axs[0].set_title(lab)

savefig4paper(fig, 'supp_expdam')



