import operator
from collections import defaultdict

from dicedps.plot.common import *
os.chdir(join(home, 'working', 'dicedps', 'sandbox'))

#df = v.load_pareto_mpi('u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#v.save_pareto(df, 'u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#df[v.get_ocols(df)]

df = load_merged(orient='max')

objlist = o.oset2labs['greg4d']
dfn, _ =get_scaled_df(df[objlist])

dfntime = dfn.loc[mtime].sort_values(o.o_min_mean2degyears_lab)
dfntime = dfntime.groupby(dfntime[o.o_min_mean2degyears_lab].round(2)).mean().reindex(np.arange(0,1,0.01)).interpolate()
dfndps = dfn.loc[mdps].sort_values(o.o_min_mean2degyears_lab)
dfndps = dfndps.groupby(dfndps[o.o_min_mean2degyears_lab].round(2)).mean().reindex(np.arange(0,1,0.01)).interpolate()

dfntime2 = []
dfndps2 = []
for j, x in enumerate(objlist):
    dfntime2.append(dfntime.sort_values(x)[o.o_min_mean2degyears_lab])
    dfndps2.append(dfndps.sort_values(x)[o.o_min_mean2degyears_lab])

cmap = plt.get_cmap('viridis')
fig, axs = plt.subplots(1,4)
#fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
for j, x in enumerate(objlist):
    yyd = dfndps2[j].values
    yyt = dfntime2[j].values
    #yy = np.linspace(0,1,len(yy))
    axs[j].imshow(np.vstack((yyd,yyt)).T, cmap=cmap, aspect='auto', origin='bottom')
    axs[j].axvline(0.5, color='w')
    #axs[i,j].set_xlim([0,10]) #len(yy)])
    #axs[i, j].set_ylim([0,1])
fig.tight_layout()


dftime = df.loc[mtime, objlist]
dfdps = df.loc[mdps, objlist]
dfntime = dfn.loc[mtime, objlist]
dfndps = dfn.loc[mdps, objlist]

df2, df1 = dfdps, dftime
domdict = {}
for xcurr, ycurr in tqdm(df1.iterrows(), total=df1.shape[0]):
    idx = ((ycurr-df2)<0).all(1)
    dfdom = df2[idx]
    if len(dfdom)>0:
        domdict[xcurr] = dfdom - ycurr
        domdict[xcurr].columns = ['d'+x for x in objlist]
        for x in objlist:
            domdict[xcurr].loc[:,x] = np.nan
        domdict[xcurr].loc[:,objlist] = dfndps[idx].loc[:,:].values

dfdom=pd.concat(domdict)

#dfdom.columns = ['d'+x for x in objlist] + objlist

dfdom_sorted.iloc[0]

fig, axs = plt.subplots(4,1, figsize=(8,8), sharex=True)
for ax, hcol in zip(axs, objlist):
    dcol = 'd'+hcol
    dfdom_sorted = dfdom.sort_values(dcol, ascending=False)
    df2deg = dfdom_sorted.groupby(np.round(dfdom_sorted[hcol],3)).first()[objlist+[dcol]]
    cmap = plt.cm.Blues
    norm = mpl.colors.Normalize(vmin=df2deg[dcol].min(), vmax=df2deg[dcol].max())
    df2plot = df2deg[objlist].copy()
    df2plot.index.name = 'a'
    df2plot.T.plot(ax=ax, alpha=1, ls='-', lw=1, legend=False, color=cmap(norm(df2deg[dcol].values)))  # **p)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap,
                                   norm=norm,
                                   orientation='vertical')
    ax.set_title(f'VOI {obj2lab2[hcol]}')
fig.tight_layout()
clp()

ax.set_ylabel(var2lab[y])

domsolcount = dfdom.groupby('idsol').count()[o.o_min_mean2degyears_lab].sort_values(ascending=False)
ymax = domsolcount.max()
xpara = range(4)
for N in [10,100,1000]:
    fig, ax = plt.subplots(1,1)
    for x,y in domsolcount.iloc[:N].items():
        ax.plot(xpara, dfdps.loc[x].values, alpha=y/ymax, **prop_list[0])


fig, axs = plt.subplots(4,1)
from pandas.plotting import parallel_coordinates
dfdom_sorted = []
for x in objlist:
    dfdom_sorted.append(dfdom.sort_values(x,ascending=False))
fig, axs = plt.subplots(4,1, figsize=(8,6))
xdim = range(0,len(objlist))
for iax, ax in enumerate(axs):
    itime, idps = dfdom_sorted[iax].iloc[0].name
    for y, p in zip([dfn.loc[mtime].loc[dfdom_sorted[iax].xs(idps, 0, 'idsol').index], dfn.loc[mdps].loc[[idps]]], prop_list):
        for xx, yy in y.iterrows():
            ax.plot(xdim, yy.values, **p)
fig.tight_layout()


coeff2op = {-1: operator.ge, 1: operator.le}

dfmed = df.loc['med']
dftime = dfmed.xs('time', 0, 'miulab')
dfdps = dfmed.xs('rbfXdX41', 0, 'miulab')[objlist]

simtime = get_sim2plot('time')
amiu = v.get_x(dftime.loc[dftime[o.o_max_rel2c_lab].idxmin()])

amiu
amiu2 = np.r_[amiu[:8],[1.2]*(47-8)]
amiu2
plt.plot(range(47), amiu, range(47), amiu2)
plot_var_cmap(simtime, ['MIU', 'TATM'], amiu)

t=simtime.dc.MIU
t.plot()
t.loc[22000:,99] = 2.7
np.maximum(t.iloc[:-10]-2,0).sum(0).mean()*5
simtime.dc.run_and_ret_objs(amiu)
simtime.dc.run_and_ret_objs(amiu2)
amiu
simtime.dc.run_and_ret_objs(1.2*np.ones(47))

df1, df2 = dfdps, dftime

for xt, yt in tqdm(dftime.iterrows(), total=dftime.shape[0]):
    dfdom = dfdps[((yt-dfdps)<0).all(1)]
    if len(dfdom)>0:
        break


dfmed = df1.median()

y = -df1[o.o_max_rel2c_lab]

x = np.linspace(y.min(), y.max(), 100)
n = len(y)

fig, axs = plt.subplots(3,4, figsize=(8,6))
for i, (obj, coeff) in enumerate(zip(objlist,[-1,-1,-1,-1])):
    for m, p in zip(['rbfXdX41',mtime], prop_list):
        ymiu = coeff*df.xs(m,0,'miulab')[obj].copy()
        ymiu.index = ymiu.index.droplevel([1,2])
        cli = 'med'
        y = ymiu.loc[cli]
        x = np.linspace(y.min(), y.max(), 100)
        x2sols = {}
        op = coeff2op[coeff]
        n = len(y)
        ys = defaultdict(list)
        for xx in x:
            currsols = y[op(y, xx)]
            ys['med'].append(currsols.count()/n)
            for cli in ['low','high']:
                ycurr = ymiu.loc[cli].loc[currsols.reset_index()['idsol'].values]
                ys[cli].append(ycurr[op(ycurr, xx)].count() / n)
        for j,cli in enumerate(['low','med','high']):
            ax = axs[j,i]
            ax.plot(x, ys[cli], label=m, **p)


#ax.plot(x, [.count()/n for xx in x], label=miu2lab[m], **p)
ax.legend()

for
df1[(df1[o.o_max_util_bge_lab]<dfmed[o.o_max_util_bge_lab]) &
(df1[o.o_min_cbgemitcost_lab]<dfmed[o.o_min_cbgemitcost_lab]) &
(df1[o.o_min_cbgedamcost_lab]<dfmed[o.o_min_cbgedamcost_lab]) &
(df1[o.o_max_rel2c_lab]<dfmed[o.o_max_rel2c_lab])].sort_values(objlist, ascending=False).iloc[0].name[1]

simdps = get_sim2plot('rbfXdX44', 100)
amiu = df.loc[mdps].xs(2171,0,'idsol')
plot_var_cmap(simdps, ['MIU', 'TATM'], amiu)
yt
dfdom
dfdps.head(10)
yt
