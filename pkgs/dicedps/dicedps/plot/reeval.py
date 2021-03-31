from dicedps.plot.common import *

inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','meeting-dicedps-20180511','figures',*x)

os.chdir(os.path.join(os.environ['HOME'], 'working/dicedps/sandbox'))
df = v.load_pareto('u1w1000*rerun.csv', objlabs=False)
y = df[o.o_max_rel2c].unstack(0)


fig, ax = plt.subplots(1,1,figsize=(4,6))
for miu, p in zip(y.index.levels[0], prop_list):
    yy = y.loc[miu]
    for col, yyy in yy.items():
        ax.scatter(-yy['u1w1000fgisstgissscauchyo4'], -yyy, **p)
ax.set_xlim([0,100])
ax.set_ylim([0,100])

miu2lab = {
    'rbfXdX4': 'Adaptive',
    'time': 'Non-adaptive'
}
from matplotlib import gridspec
#highcol = 'u1w1000fgissTgissOchengschyleko10'
smed_dict = {}
csfocus = ['u1w1000fgissTgissOchengschyleko4', 'u1w1000fgisstgissscauchyo4', 'u1w1000fgissThadcrutscauchyo10']
cslow, csmed, cshigh = csfocus
ncfiles = []
for ihigh, highcol in enumerate(csfocus):
    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 3], figure=fig)
    ax = plt.subplot(gs[0])
    for i, dist in enumerate(csfocus):
        if not dist in csfocus:
            continue
        if dist == highcol:
            color = 'k'
        else:
            color = 'gray'
        ncfile = inrootdir('dicedps','data',(dist.replace('u1w1000','brick_').replace('T','_T').replace('tgi','_tgi')
               .replace('sch','_sch').replace('sinf','_sinf').replace('scau','_scau').replace('slog','_slog')
               .replace('o4','_o4').replace('o10','_o10')))+'.nc'
        ncfiles.append(ncfile)
        sdata = u.ncbrick2pandas(ncfile, columns=None)['S']
        smed = sdata.median()
        smed_dict[dist] = smed
        sq25 = sdata.quantile(0.25)
        sq75 = sdata.quantile(0.75)
        ax.errorbar(i, smed, np.array([[sq75-smed],[smed-sq25]]), color=color, lw=2, marker='o')
    ax.set_ylabel('Climate sensitivity [K]')
    ax.get_xaxis().set_ticks([])
    ax.set_xlim([-1,3])

    #pd.Series(smed_dict).argmax()

    ax = plt.subplot(gs[1])
    for miu, p in zip(y.index.levels[0], prop_list):
        yy =  df[o.o_max_rel2c].unstack(0).loc[miu]
        ax.scatter(-yy['u1w1000fgisstgissscauchyo4'], -yy[highcol], label=miu2lab[miu], **p)
    ax.legend(loc='best')
    ax.set_ylim([0,100])
    ax.set_xlabel('Nominal Reliability 2C (%)')
    ax.set_ylabel('Reliability 2C (%)')

    """
    ax = plt.subplot(gs[2])
    for miu, p in zip(y.index.levels[0], prop_list):
        yy =  df[o.o_min_npvmitcost].unstack(0).loc[miu]
        ax.scatter(yy['u1w1000fgisstgissscauchyo4'], yy[highcol], **p)
    """

    ax = plt.subplot(gs[2])
    for miu, p in zip(y.index.levels[0], prop_list):
        yy = df.loc[highcol].loc[miu]
        ax.scatter(yy[o.o_min_npvmitcost], -yy[o.o_max_rel2c], **p)
    ax.set_ylim([0,100])
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel('NPV mitigation cost (% GPD)')
    fig.tight_layout()
    fig.savefig(inplot(f'reeval{ihigh}.pdf'))

clp()
fig, axs = plt.subplots(4,5,figsize=(12,12))
for i, (ax, col) in enumerate(zip(axs.flat, y.columns)):
    if i==2:
        print(col)
    for miu, p in zip(y.index.levels[0], prop_list):
        yy = df[o.o_max_rel2c].unstack(0).loc[miu]
        #yy =  df[o.o_min_npvmitcost].unstack(0).loc[miu]
        ax.scatter(-yy['u1w1000fgisstgissscauchyo4'], -yy[col], **p)
    ax.set_title(col)
clp()

inplot = lambda x: os.path.join(os.environ['HOME'],'working','presentation-egu2018-dice-dps','figures', x)

dc = h.args2dice(f'-m time -o greg4 -u 1 -w 1 -e 2100 -C ka18')

dc0 = dc.run(np.zeros(17))
dc0.EIND.plot()
#df = xr.open_dataset(inrootdir('sandbox/reeval.nc')).to_dataframe()
df = pd.read_csv(inrootdir('sandbox/reeval.csv'), index_col=[1,0,2])
ocost = 'MIN_NPVMITCOST'
orel = 'MAX_REL2C'
df['cheap'] = np.maximum(df[ocost]-3,0)
df['cheap_bool'] = df[ocost]<3
df['rel_bool'] = (df['MAX_REL2C'] >= 66)

fig, axs = plt.subplots(1,3, figsize=(8,3))
def anynull(x):
    return pd.isnull(x).any()
def countzeros(x):
    return (np.isclose(x,0)).astype(int).sum()
def df2matrix(x,miu):
    return x.loc[miu].unstack().T
dfbydist_null = df.groupby(['miu', 'mu', 'sigma'])['cheap'].apply(anynull)
dfbydist_count = df.groupby(['miu', 'mu', 'sigma'])['cheap'].apply(countzeros)
dfbydist_count.to_frame().to_html('prova.html')
sb.heatmap(data=df2matrix(dfbydist_count, mtime), ax=axs[0], cmap='Reds',mask=df2matrix(dfbydist_null, mtime))
axs[0].set_title('Open loop\n% of Rel2C | MitCost<3%', fontsize=10)
axs[1].set_title('Closed loop\n% of Rel2C | MitCost<3%', fontsize=10)
axs[0].invert_yaxis()
sb.heatmap(data=df2matrix(dfbydist_count, mdps), ax=axs[1], cmap='Blues',mask=df2matrix(dfbydist_null, mdps))
axs[1].invert_yaxis()
for ax in axs[:1]:
    ax.set_xlabel('ECS mu')
    ax.set_ylabel('ECS sigma')
axs[2].clear()

fig, axs = plt.subplots(1,2,figsize=(8,5))
ax=axs[0]
for dist in df.index.levels[0]:
    h = u.ClimateSensitivityRV(dist).plot(ax=ax, color='k', lw=2, alpha=0.5)
ax.set_xlim([0.5,8.5])
#ax.set_ylabel('Probability density')
ax.set_xlabel('Climate sensitivity (degC)')
ax.set_title(f'{len(df.index.levels[0])} alternative CS prob. density\ndistributions from IPCC AR5', fontsize=10) #legend(h,[])

df.groupby(['miu', 'rel2c'])['cheap_bool'].first()
dfcheap = df.groupby(['miu', 'rel2c'])['cheap_bool'].sum().unstack().T
dfcheap.columns = [cdps, copen]
ax=axs[1]
colors = np.array(sb.color_palette())[[0, 3]]
dfcheap.plot(ax=ax, color=colors)
ax.set_title('# CS distributions s.t. solutions with\nmitigation cost < 3% exist for a given 2C reliability',fontsize=10)
ax.set_xlim([60,81])
ax.set_ylim([0,21])
ax.set_xlabel('Solutions ordered by increasing nominal Reliability 2C goal')
fig.tight_layout()
sb.despine(fig)
#fig.savefig(inplot('fig_reeval.png'), dpi=200)
fig.savefig(inplot('fig_reeval.pdf'))


#######
def f(x):
    return pd.Series([x.min(),x.max(),x.mean(),x.loc['olson_informPrior'].iloc[0]],index=['min','max','mean','olson'])
costdfstats = df.groupby(['miu', 'rel2c'])[ocost].apply(f).unstack()
rel2cdfstats = df.groupby(['miu', 'rel2c'])[orel].apply(f).unstack()


fig, axs = plt.subplots(1,2,figsize=(8,4),sharey=False)
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for miu, p in zip([mdps,mtime], prop_cycle):
    mdfstats = df.xs(miu,0,'miu').reset_index()
    axs[0].scatter(mdfstats['rel2c'],mdfstats[orel], alpha=0.5,**p)
    axs[1].scatter(mdfstats['rel2c'], mdfstats[ocost], alpha=0.5, **p)
    #axs[0].fill_between(mdfstats.index, mdfstats['min'], mdfstats['max'], alpha=0.5, **p)
for ax in axs:
    ax.set_xlabel('Solutions by nominal reliability 2C')
axs[0].set_ylabel('Realized reliability 2C')
axs[1].set_ylabel('Realized mitigation cost')
fig.tight_layout()
fig.savefig(incloud('fig_reeval_relcost.png'),dpi=200)

mdfstats
######
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for miu, p in zip([mdps,mtime], prop_cycle):
    mdfstats = dfstats.loc[miu]
    axs[1].fill_between(mdfstats.index, mdfstats['min'], mdfstats['max'], alpha=0.5, **p)


sb.factorplot(x='rel2c',y=ocost,hue='miu',x_order= data=df.loc['olson_informPrior'].reset_index(),kind='point')
df.groupby(['miu', 'rel2c'])['rel_bool'].sum().unstack().T.plot(ax=axs[1])
axs[1].set_title('# ECS distributions\nRel>66%',fontsize=10)
