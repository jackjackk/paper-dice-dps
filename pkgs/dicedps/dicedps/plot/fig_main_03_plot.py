from dicedps.plot.common import *


#%% load data

vlist = [
    ldf_damfunc,
    ldf_csdist,
    ldfthinned_mitcost,
    ldf_interp_mitcost,
    ldf_thinned_diffs
]

data = {}
for vcurr in vlist:
    data[vcurr] = pd.read_parquet(inoutput('dicedps', f'{vcurr}.dat'))


#%% init figure grid

clp()
fig = plt.figure(figsize=(w2col*1.5,hhalf*1.7))
gs = GridSpec(6, 6)
ax_damfunc = plt.subplot(gs[:3, :2])
ax_cs = plt.subplot(gs[3:,:2])
ax_troff: plt.Axes = plt.subplot(gs[:3,2:6])
ax_diff = plt.subplot(gs[3:,2:6])
#ax_diff2 = plt.subplot(gs[:5,5:])

prop_list2 = [prop_list[i] for i in [2,4,3,5,6]]+[{'color':'0.5'}]


ldam = ['Nordhaus (Nominal)', 'Weitzman', 'Goes']


#%% damage functions

ax_damfunc.clear()
for i, l in enumerate(ldam2):
    y = data[ldf_damfunc][l]
    ax_damfunc.plot(y.index, y.values, label=l, **prop_list2[i])
hleg = ax_damfunc.legend(title=ldamfunc_lab)
plt.setp(hleg.get_title(),fontsize='small')
ax_damfunc.set_xlabel('Temperature (°C)')
ax_damfunc.set_ylabel('GDP loss (%)')


#%% climate sensitivity

ax_cs.clear()
for cslab, p in zip(lcslabs, prop_list2):
    sb.distplot(data[ldf_csdist][cslab], label=cslab, ax=ax_cs, kde=False, norm_hist=True, hist_kws={'alpha': 0.5},
                **p)
hleg = ax_cs.legend(title='Climate sensitivity')
plt.setp(hleg.get_title(), fontsize='small')
ax_cs.set_xlabel('Climate sensitivity (K)')
ax_cs.set_ylabel('Probability Density Function')


#%% trade-offs

scen2lab = {
    'low1': 'Low CS + Low DF',
    'med2': 'Medium CS + Medium DF',
    'high3': 'High CS + High DF',
    'high2': 'High CS + Medium DF',
    'med3': 'Medium CS + High DF',
    'med1': 'Nominal',
}

stidy = lambda x: x.replace('CS', 'climate sensitivity').replace('DF', 'damage function').replace(' + ', '\n')

scen2off = {
    'med1':-10,
    'med2':-20,
    'high2':20,
    'med3':-25,
    'low1':10,
}


ax_troff.clear()
ax_diff.clear()
#ax_diff2.clear()

hs = []
ydict = defaultdict(dict)
yotherdict = defaultdict(dict)
scenlist = ['low1','med2','high3','high2','med3','med1']
hsols = {}
hinter = {}
hs_diff = []

for scen, p in zip(scenlist, prop_list2):
    scen_cli = scen[:-1]
    scen_df = scen[-1]
    dfclidf_mitcost = (data[ldfthinned_mitcost]
                       .xs(scen_cli, 0, 'climcalib')
                       .xs(scen_df, 0, 'damfunc'))
    # plot markers
    for miu, s, mk, ls, fs in zip([mtime, mdps], [20,50], ['s','.'], ['--', '--'], ['none',p['color']]):
        dfcurr = dfclidf_mitcost.xs(miu, 0, 'miulab')
        hsols[miu] = ax_troff.scatter(dfcurr[f3xcol], dfcurr[f3ycol],
                                      s=s, marker=mk, facecolors=fs, **p)
    # plot interpolated fill
    y = data[ldf_interp_mitcost].loc[scen]
    hvoi = ax_troff.fill_between(y.loc[mdps].index,
                          y.loc[mdps,f3ycol],
                          y.loc[mtime,f3ycol],
                          alpha=0.3, **p)
    # plot lines
    for idx, sol in dfclidf_mitcost.unstack('miulab').iterrows():
        hline = ax_troff.plot([sol[f3xcol][mtime], sol[f3xcol][mdps]],
                              [sol[f3ycol][mtime], sol[f3ycol][mdps]],
                              alpha=0.5,
                              **p)[0]
    # plot diff
    y = data[ldf_thinned_diffs].loc[scen]
    #ax_diff2.plot(sdeltax.index, sdeltax.values, **p)
    hs_diff.append(ax_diff.plot(y['idy'], y['dy'], **p)[0])

    # plot labels
    # ax_troff.annotate(s=stidy(scen2lab[scen]),
    #                   xy=(dfcurr[f3xcol].iloc[-1],
    #                       dfcurr[f3ycol].iloc[-1]),
    #                   xytext=(0,scen2off.get(scen, 15)),
    #                   textcoords='offset points', ha='right', va='bottom', fontsize=8)


ax_troff.legend([hsols[mtime],
                 hsols[mdps],
                 #hinter[mtime],
                 #hinter[mdps],
                 #hline,
                 hvoi],
                ['Non-adaptive strategies',
                 'Adaptive strategies',
                 #'Interpolated Pareto-front',
                 #'Matching strategies with\nthe same nominal mitigation cost',
                'Regret of non-adaptive\nstrategies'], ncol=2)

xlab = obj2lab2[f3xcol].replace('\n', ' ')
ax_troff.set_xlabel(xlab)
ax_troff.set_ylabel(obj2lab2[f3ycol])

#%%

for ax in [ax_troff,]: # ax_diff2]:
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.5,1,5,10,50]))
    ax.yaxis.set_minor_locator(ticker.FixedLocator([0.6,0.7,0.8,0.9,
                                                    2,3,4,6,7,8,9,
                                                    20,30,40]))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter(''))

ax_diff.legend(hs_diff,
               [stidy(scen2lab[scen]) for scen in scenlist])

ax_troff.set_ylim([0.5,50])

ax_diff.set_xlabel(xlab)
ax_diff.set_ylabel('Regret of non-adaptive strategies\n95th-percentile Damage cost (% CBGE)')


#%%

hletters = []
for i, ax in enumerate([ax_damfunc, ax_cs, ax_troff, ax_diff]):  # ax_diff2,
    sb.despine(fig, ax)
    hletters.append(ax.text(0.95, 1., string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))
fig.tight_layout()

savefig4paper(fig, 'main_03')
