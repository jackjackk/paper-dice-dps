from dicedps.plot.common import *

### BARPLOT OF INDICATORS ###
indlist_all = ['Hypervolume', 'Generational Distance','Inverted GD','Spacing','Epsilon Indicator','Max Pareto Front Error']
indlist = ['Hypervolume', 'Generational Distance', 'Epsilon Indicator']

flist = np.array(indata('bymiu*.finalmetrics'))
miulist = np.array([filename2miu(f) for f in flist])

df = pd.concat([pd.read_csv(f, sep= ' ', header=0) for f in flist], keys=miulist).reindex_axis(miu_nice_order, axis=0, level=0)
df.columns = indlist_all
df = df[indlist].stack().reset_index()
df.columns = ['Control','Seed','Indicator','Value']
hv=df[df.Indicator=='Hypervolume'].groupby('Control')['Value'].mean()
hv.loc['DPS(T,dT|4)']/hv.loc['Open loop']-1
clp()
cols = np.array(sb.color_palette())[[3,1,2,0]]
g = sb.factorplot(row='Indicator',x='Value',y='Control',data=df,kind='bar',size=2,aspect=16/9*3/2, sharex=False, orient='h', palette=cols)
for ax,ind in zip(g.axes.flat, indlist): ax.set_title(ind); ax.set_xlabel(''); ax.set_ylabel('')
g.fig.tight_layout(pad=1.01)
g.fig.savefig(incloud(os.path.join('plots', 'fig_indicators.png')), dpi=200)


### INDICATORS OVER NFE ###
flist = np.array(glob(incloud(os.path.join('data', '*.metrics'))))
fkeys = [tuple(np.array(os.path.basename(f).split('.')[0].split('_'))[[1,3,5]]) for f in flist]
miu_nice_order = ['Open loop', 'DPS(T|4)', 'DPS(T,dT|1)', 'DPS(T,dT|4)','DPS(T,dT|6)']
df = pd.concat([pd.read_csv(f, sep= ' ', header=0, index_col=0) for f in flist], keys=fkeys).rename_axis(miumap2, axis=0).reindex_axis(miu_nice_order, axis=0, level=0)
df.columns = ['Generational Distance', 'Epsilon Indicator', 'Hypervolume', 'Contribution']
a = df['Hypervolume']
df = df['Hypervolume'].reset_index()
#df.columns = ['Control','Seed','TOTNFE','NFE','Hypervolume']
df.columns = ['Control','Seed','NFE','Hypervolume']
dff = df.groupby(['Control','NFE'])['Hypervolume'].mean()
dff.index.levels[0]
fig, ax = plt.subplots(1,1,figsize=(4,5))
for miu, color in zip(miu_nice_order, np.array(sb.color_palette())[[3,1,2,0,4]]):
    ax.semilogx(dff.loc[miu], color=color, lw=2, label=miu)
ax.legend()
sb.despine(fig)
ax.set_xlabel('NFE')
ax.set_ylabel('Hypervolume')
ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xlim(ax.get_xlim())
fig.tight_layout()
ax2.get_xticks()

tdf=pd.read_csv(indata('timings.csv')[0],sep='\s',index_col=None,header=None)
tdf.columns=['nfe','time']
tdf.groupby('nfe')['time'].mean().mul(20).loc[[100000,1000000]]
ax2.set_xticklabels(['','','~10 hours','~100 hours'])
ax2.set_xlabel('Computational time')
g = sb.factorplot(x='NFE',y='Hypervolume',hue='Control',data=df,kind='point',aspect=16/9/2,size=6, legend_out=False, palette=cols)
for i, x in enumerate(g.axes.flat[0].get_xticklabels()): x.set_visible(not i%5)
fig.savefig(inplot('fig_hypervolume_vs_nfe.png'), dpi=200)

mius=a.index.levels[0].tolist()
fig, ax = plt.subplots(1, 1)
for x,y in a.unstack('#NFE').iterrows():
    ax.plot(y.index, y.values, color=cols[mius.index(x[0])])

a.xs('nfe4000000',0,1)

"""
g.fig.autofmt_xdate(rotation=90, ha='center')
g.axes.flat[0].set_ylabel('')

for ax,ind in zip(g.axes.flat, indlist): ax.set_xlabel(''); fig.autofmt_xdate()
sb.barplot()
"""