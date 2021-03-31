from dicedps.plot.common import *

df = load_rerun('greg4h', orient='min')

scol = 'Damage + Mitigation cost'
df[scol] = df[o.o_min_cbgedamcost_lab]+df[o.o_min_cbgemitcost_lab]
N = 20
xcol = o.o_min_mean2degyears_lab #o.o_min_q95maxtemp_lab
#ycol = o.o_min_q95damcost_lab
ycol = scol
mcol = o.o_min_cbgemitcost_lab
xall = df[xcol]
yall = df[ycol]
x = np.geomspace(xall.min(), xall.max(), N)
y = np.geomspace(yall.min(), yall.max(), N)
y = np.geomspace(0.2,5,N)
#x = np.linspace(2,5,N)
x = np.geomspace(1,100, N)

X, Y = np.meshgrid(x,y)
Z = np.zeros((N, N))


"""
df[o.oset2labs['greg4d']].describe()
fig = plt.figure(figsize=(6, 4))

grid = AxesGrid(fig, 111,
                nrows_ncols=(3, 3),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
"""
cmap = plt.get_cmap('viridis')

xcol = o.o_min_mean2degyears_lab
ycol = o.o_min_cbgedamcost_lab
ids2plot = {}
for miu in [mdps,mtime]:
    dfnom = df.loc['med1'].loc[miu]
    dfnom_sorted = dfnom.sort_values(o.o_min_cbgemitcost_lab).reset_index()
    ids2plot[miu] = dfnom_sorted.groupby(np.round(dfnom_sorted[xcol],0)).first()['idsol']

fig1, axs1 = plt.subplots(3,3, sharex=True, sharey=True, figsize=(8,6))
for axs, m, p in zip([axs1,axs1], [mdps,mtime], prop_list):
    for irow, sdam in enumerate('321'):
        for icol, scs in enumerate(['low','med','hig']):
            dfcurr = df.loc[f'{scs}{sdam}'].loc[m].loc[ids2plot[m]]
            axs[irow,icol].scatter(dfcurr[xcol],dfcurr[o.o_min_cbgedamcost_lab], label=miu2lab[m], s=3, **p)
            if irow==0 and icol==0:
                axs[irow,icol].legend()
            axs[irow, icol].scatter(dfcurr[xcol], dfcurr[o.o_min_q95damcost_lab], label=None, s=3, **p)
axs[2,1].set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
axs[1,0].set_ylabel('Mean & 95th-percentile Damage cost (% CBGE)')
#axs[0,0].set_yscale('log')
#axs[0,0].set_xscale('log')




for miu in [mtime,mdps]:

    Z = np.zeros((N, N))
    for i in range(N):
        if not bcontinue:
            break
        for j in range(N):
            if not bcontinue:
                break
            # Z[i,j] = len(dfcurr[(dfcurr[xcol]<=X[i,j]) & (dfcurr[o.o_min_q95damcost]<=Y[i,j]) & (dfcurr[ycol]<=Y[i,j]) & (dfcurr[mcol]<=1)])
            dfsols = dfnom[(dfnom[xcol] <= X[i, j]) & (
                    (dfnom[ycol]) <= Y[i, j])]
            Z[i, j] = len(dfsols)
            if Z[i,j] == 1:
                bcontinue = False


fig1, axs1 = plt.subplots(3,3, sharex=True, sharey=True, figsize=(8,8))
#fig2, axs2 = plt.subplots(3,3, sharex=True, sharey=True, figsize=(8,8))
zmax = 0
for axs, m, p in zip([axs1,axs1], [mdps,mtime], prop_list):
    for irow, sdam in enumerate('321'):
        for icol, scs in enumerate(['low','med','hig']):
            dfcurr = df.loc[f'{scs}{sdam}'].loc[m]
            dfnom = df.loc['med1'].loc[m]
            bcontinue = True
            Z = np.zeros((N, N))
            for i in range(N):
                if not bcontinue:
                    break
                for j in range(N):
                    if not bcontinue:
                        break
                    #Z[i,j] = len(dfcurr[(dfcurr[xcol]<=X[i,j]) & (dfcurr[o.o_min_q95damcost]<=Y[i,j]) & (dfcurr[ycol]<=Y[i,j]) & (dfcurr[mcol]<=1)])
                    Z[i, j] = min(1, len(dfcurr[(dfcurr[xcol] <= X[i, j]) & (
                            (dfcurr[ycol]) <= Y[i, j]) & (
                        dfnom[xcol] <= X[i, j]) & (
                                (dfnom[ycol]) <= Y[i, j])]))
                    zmax = max(Z[i,j], zmax)
                    #if (i > 0) and (j > 0):
                    #    if (Z[i,j-1]>0) and (Z[i-1,j-1]>0) and (Z[i-1,j]>0):
                    #        bcontinue = False
            axs[irow,icol].contour(X, Y, Z, levels=[1], colors=p['color'])
            axs[irow, icol].set_yscale('linear')
            axs[irow, icol].set_xscale('linear')
            #axs[irow, icol].pcolormesh(X, Y, Z, cmap=cmap) # levels=[1], colors=p['color'])

axs[2,1].set_xlabel('95th-percentile max temperature (K)')
axs[1,0].set_ylabel('95th-percentile damage cost (% CBGE)')



cbar = mpl.colorbar.ColorbarBase(ax=grid.cbar_axes[0], cmap=cmap,
                               spacing='uniform',
                               orientation='vertical',
                               extend='neither')
cbar.set_ticks([0,0.5,1])
cbar.ax.set_yticklabels([f'{x:.0f}' for x in [0,zmax//2,zmax]])
cbar.ax.set_ylabel('# solutions',rotation=-90, labelpad=15)

sdam
df.loc['low1']

dfnom = df.loc['med1']
dfnom.name = 'min'
fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
plot_objective_pairs(dfnom, orows=[o.o_min_mean2degyears_lab], ocols=[o.o_min_miu2020_lab, o.o_min_miu2030_lab, o.o_min_miu2050_lab], axs=axs, transpose=True)


df[o.o_min_mean2degyears_lab]
