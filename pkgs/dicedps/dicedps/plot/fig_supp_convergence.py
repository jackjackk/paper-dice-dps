from dicedps.plot.common import *

lhv = '+hypervolume'

dmetrics = {}
for fmetric in glob(inscratch('dicedps', '*_metrics.csv')):
    fspec = fmetric.split('_')
    dmetrics[(fspec[1],fspec[-2])] = pd.read_csv(fmetric, sep=' ', index_col=0)


dfm = pd.concat(dmetrics, names=['miulab','seed','nfe'])
dfmr = dfm.reset_index()
dfmr['nfe'] /= 1e6
dfmr.head()
dfm = dfmr.set_index(keys=dfmr.columns[:3].tolist())

fig, ax = plt.subplots(1,1,figsize=(w2col,hhalf))
for miu, pro in zip(miulist, prop_list):
    for s in dfm.index.levels[1]:
        dfm.loc[f'm{miu}'].loc[s][lhv].plot(ax=ax, logx=False, **pro)
ax.legend([plt.Line2D((0,1),(0,0), ls='-', lw=2, **pro) for pro in prop_list],
          [f'Single seed, {miu2lab[miu]}' for miu in miulist],
          loc='lower right')
ax.set_xlabel('Number of function evaluations [Million]')
ax.set_ylabel('Hypervolume')
ax.set_ylim([0.55,0.65])
#sb.despine(fig)

ax2 = ax.twiny()
#ax2.set_xscale('log')
ax2.set_xlim(ax.get_xlim())
cputime= (dfm.groupby('nfe')['ElapsedTime']
    .mean().round()
    .div(3600/400.)
    .astype(int)
    .reset_index()
    .set_index('ElapsedTime')
    .reindex(range(5100))
    .interpolate().loc[[500,1000,2000,5000]]) #.loc[np.arange(0,4.1,0.1)].interpolate().loc[ax2.get_xticks()]/3600*200
ax2.set_xticks(cputime['nfe'].values)
ax2.set_xticklabels([f'{x:.0f}' for x in cputime['nfe'].index])
ax2.set_xlabel('CPU time [Hour]')


savefig4paper(fig, 'supp_convergence')