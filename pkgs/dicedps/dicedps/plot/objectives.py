from dicedps.plot.common import *

inplot = lambda x: os.path.join(os.environ['HOME'],'working','presentation-egu2018-dice-dps','figures', x)

### OBJECTIVE PAIRS ###
flist = indata('bymiu_*.ref')
miulist = list(map(filename2miu, flist))
df = pd.concat(
    [pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist],
    keys=miulist,
    names=['Control', 'Idsol'])

#df = df.loc[].reset_index().drop('Idsol', axis=1)
df.columns = olist
df[orel2c] = df[orel2c].mul(-1)
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
colors = np.array(sb.color_palette())[[0, 3]]
#prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for obj1, obj2, ax in zip([omit, omit, omit, orel2c],
                          [orel2c, odam, outil, odam], axs.flat):
    for miu, p in zip([cdpstdt4, copen], colors):
        ax.scatter(df.loc[miu, obj1], df.loc[miu, obj2], alpha=1, s=2, color=p)
        ax.scatter(
            o2best[obj1](df.loc[miu, obj1]),
            o2best[obj2](df.loc[miu, obj2]),
            color='k',
            marker='D')
    ax.set_xlabel(obj1)
    ax.set_ylabel(obj2)
hdps = plt.Line2D((0, 1), (0, 0), color=colors[0], lw=2)
hopen = plt.Line2D((0, 1), (0, 0), color=colors[1], lw=2)
hbest = plt.Line2D(
    (0, ), (0, ), color="white", marker='D', markerfacecolor='k')
hthres = plt.Rectangle((0, 0), 1, 1, fill=False)
fig.tight_layout(rect=[0, 0, 1., 0.85])
l = axs[0,0].legend(
    [hdps, hopen, hbest, hthres], [cdps, copen, 'Preferred value'],
    bbox_to_anchor=(0, 1.15, 1, 0.2),
    loc="lower left",
    borderaxespad=0,
    ncol=4)
sb.despine(fig)
fig.savefig(inplot('fig_obj_pairs.png'), dpi=200)
fig.savefig(inplot('fig_obj_pairs.pdf'))

### PARALLEL PLOT ###
flist = indata('bymiu_*.ref')
miulist = list(map(filename2miu, flist))
df = pd.concat(
    [pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist],
    keys=miulist,
    names=['Control', 'Idsol']).loc[[copen, cdpstdt4]]
for obj in olist:
    df[obj] = df[obj].mul(-1)
idx_within_thres = ((df[orel2c] >= 66) & (df[omit] > -3) & (df[odam] > -0.6))
idx_rel2c_odd = (df[orel2c].astype(int) % 2)
idx_bymiu = {cdpstdt4: idx_rel2c_odd, copen: ~idx_rel2c_odd}
ndf = df.copy()
scaled_values = scaler.fit_transform(df)
ndf.loc[:, :] = scaled_values
#ndf['Reliability_2C'] = 1-ndf['Reliability_2C'] #.round(2).mul(100).astype(int)
#ndf.groupby('Control')['Reliability_2C'].describe()
ndf2dec = ndf.round(2).drop_duplicates()
y_lowbound = scaler.transform([[66, 0, -3, 0]])[0]
idx_within_thres_2dec = ndf2dec.iloc[:, 0].mul(0).add(1).astype(bool)
for obj, low in zip(olist, y_lowbound):
    idx_within_thres_2dec = idx_within_thres_2dec & (ndf2dec[obj] >= low)
idx_rel2c_odd_2dec = (ndf2dec[orel2c].mul(100).astype(int) % 2)
idx_bymiu_2dec = {cdpstdt4: idx_rel2c_odd_2dec, copen: ~idx_rel2c_odd_2dec}

colors = np.array(sb.color_palette())[[0, 3]]
clp()

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
cmap_grays = mpl.cm.get_cmap('Greys')
for cmap_lab, miu2fill, col in zip(['winter', 'autumn'], [cdpstdt4, copen],
                                   colors):
    cmap = mpl.cm.get_cmap(cmap_lab)
    for x, y in tqdm(
            ndf2dec[(~idx_within_thres_2dec)
                    & idx_bymiu_2dec[miu2fill]].loc[miu2fill].iterrows()):
        #for x, y in tqdm(ndf[(~idx_within_thres)].loc[miu2fill].iterrows()):
        #col=cmap_grays((y['Reliability_2C']))
        hopen = ax.plot(y.values, color='gray', ls='-', lw=1, alpha=0.05)
    for x, y in ndf[idx_within_thres].loc[miu2fill].iterrows():
        #col=  #cmap((y['Reliability_2C']))
        hopen = ax.plot(y.values, color=col, ls='-', lw=2, alpha=0.6)
ax.add_patch(
    mpl.patches.Rectangle(
        (-0.05, y_lowbound[0]),  # (x,y)
        0.1,  # width
        1 - y_lowbound[0],  # height
        fill=False,
        zorder=100.))
ax.add_patch(
    mpl.patches.Rectangle(
        (2 - 0.05, y_lowbound[2]),  # (x,y)
        0.1,  # width
        1 - y_lowbound[2],  # height
        fill=False,
        zorder=100.))
for x, y in zip(range(4), [1, 1, 1, 1]):
    ax.scatter(x, y, color='k', marker='D', s=30, zorder=200)
ax.annotate(
    '>66',
    xy=(0, (y_lowbound[0])),
    xycoords='data',
    xytext=(0, -5),
    textcoords='offset points',
    horizontalalignment='center',
    verticalalignment='top')
nsol_open = df[idx_within_thres].loc[copen].shape[0]
nsol_open_tot = df.loc[copen].shape[0]
nsol_dps = df[idx_within_thres].loc[cdpstdt4].shape[0]
nsol_dps_tot = df.loc[cdpstdt4].shape[0]
ax.annotate(
    f'{nsol_open}\nsolutions',
    xy=(0, (y_lowbound[0])),
    xycoords='data',
    xytext=(90, -90),
    textcoords='offset points',
    fontweight='bold',
    horizontalalignment='center',
    verticalalignment='top',
    color=colors[1])
ax.annotate(
    f'{nsol_dps}\nsolutions',
    xy=(2, (y_lowbound[2])),
    xycoords='data',
    xytext=(-55, 60),
    textcoords='offset points',
    fontweight='bold',
    horizontalalignment='center',
    verticalalignment='top',
    color=colors[0])
ax.annotate(
    '<3',
    xy=(2, (y_lowbound[2])),
    xycoords='data',
    xytext=(0, -5),
    textcoords='offset points',
    horizontalalignment='center',
    verticalalignment='top')

ax.set_xticks(range(4))
ax2 = ax.twiny()
ax.set_xlim([-0.1, 3.1])
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(range(4))
#ax.set_xticklabels([f'{x:.1f}\n{s}' for x,s in zip(scaler.data_min_,['Reliability\n2C goal (% SOWs)','Utility\nLoss (% BAU)','Mitigation\nCost (% GDP)','Damage\nCost (% GDP)'])])
#ax2.set_xticklabels([f'{x:.1f}' for x in scaler.data_max_])
ax.set_xticklabels([])
ax2.set_xticklabels([
    f'{s}' for s in [
        'Reliability\n2C goal (% SOWs)', 'Utility\nLoss (% BAU)',
        'Mitigation\nCost (% GDP)', 'Damage\nCost (% GDP)'
    ]
])
ax.tick_params(left=False)
ax.tick_params(labelleft=False)

hdps = plt.Line2D((0, 1), (0, 0), color=colors[0], lw=2)
hopen = plt.Line2D((0, 1), (0, 0), color=colors[1], lw=2)
hbest = plt.Line2D(
    (0, ), (0, ), color="white", marker='D', markerfacecolor='k')
hthres = plt.Rectangle((0, 0), 1, 1, fill=False)
l = ax.legend(
    [hdps, hopen, hbest, hthres],
    [cdps, copen, 'Preferred value', 'Thresholds'],
    bbox_to_anchor=(0, 1.2, 1, 0.2),
    loc="lower center",
    borderaxespad=0,
    ncol=4)
fig.tight_layout(rect=[0, 0, 1., 0.9])
fig.savefig(inplot('fig_parallel_plot.png'), dpi=200)

clp()

### 3D PLOTS ###
flist = indata('bymiu_*.ref')
miulist = list(map(filename2miu, flist))
dfboth = pd.concat(
    [pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist],
    keys=miulist,
    names=['Control', 'Idsol']).loc[[copen, cdpstdt4]]
s = scaler.fit(dfboth[[outil]])

fig = plt.figure(figsize=(8, 5))
ims = []
for i, ang in zip(range(1, 4), [-30, -150]):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    for j, ccurr, cmap_name in zip(
            range(2), [cdpstdt4, copen], ['winter', 'Reds']):
        df = dfboth.loc[ccurr]
        x = df[omit].values
        y = df[odam].values
        z = -df[orel2c].values
        c = s.transform(df[[outil]])[:, 0]
        cmap = mpl.cm.get_cmap(cmap_name)
        ims.append(ax.scatter(xs=x, ys=y, zs=z, s=5, c=c, cmap=cmap))
        plt.axis('on')
        ax.set_xlabel('Mitigation\nCost (% GDP)')
        ax.set_ylabel('Damage\nCost (% GDP)')
        if i == 2:
            ax.set_zlabel('Reliability 2C goal (% SOWs)', labelpad=1)
            #ax.set_zticklabels([])
        #ax.set_yticklabels([])
        #ax.set_zticklabels([])
        ax.set_zlim([0, 85])
        ax.set_ylim([0.5, 1.3])
        ax.set_xlim([0, 3.7])
        ax.scatter(
            ax.get_xlim()[0],
            ax.get_ylim()[0],
            ax.get_zlim()[1],
            color='k',
            marker='D')
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        #[t.set_visible(False) for t in ax.xaxis.get_major_ticks()]
        #[t.set_visible(False) for t in ax.yaxis.get_major_ticks()]
        #[t.set_visible(False) for t in ax.zaxis.get_major_ticks()]
        ax.view_init(elev=15, azim=ang)
fig.tight_layout(rect=[0, 0, 0.84, 1.])  # left, bottom, width, height
cbar_im1a_ax = fig.add_axes([0.85, 0.15, 0.025, 0.6])
cbar_im1a = fig.colorbar(ims[0], cax=cbar_im1a_ax, ticks=np.linspace(0, 1, 6))
cbar_im1a.ax.set_title('Closed\nloop')
cbar_im1a.ax.set_yticklabels(
    [f'{x:.1f}' for x in np.linspace(s.data_min_, s.data_max_, 6)])
cbar_im2a_ax = fig.add_axes([0.93, 0.15, 0.025, 0.6])
cbar_im2a = mpl.colorbar.ColorbarBase(
    ax=cbar_im2a_ax,
    cmap=cmap,
    spacing='uniform',
    orientation='vertical',
    extend='neither')
cbar_im2a.ax.set_yticklabels([])
cbar_im2a.ax.set_ylabel('Utility', rotation=-90, labelpad=15)
cbar_im2a.ax.set_title('Open\nloop')
cbar_im2a_ax.yaxis.set_ticks_position('left')
cbar_im2a.get_clim()
fig.savefig(inplot('fig_dinosaur.png'), dpi=200)

clp()
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cmap = mpl.cm.get_cmap('greys')
norm = mpl.colors.normalize(
    vmin=-dfcurr['obj0'].max(), vmax=-dfcurr['obj0'].min())
for i, y in dfcurr.iterrows():
    ax.plot(y[ixcols].values, color=cmap(norm(-y['obj0'])))
cb = mpl.colorbar.colorbarbase(
    ax=cax,
    cmap=cmap,
    norm=norm,
    spacing='uniform',
    orientation='vertical',
    extend='neither')

plt.show()
clp()
plt.plot(
    ndfrel.xs(cdpstdt4, 1, 1).iloc[::10].reset_index().values.T,
    color=cmap(ndfrel.xs(copen, 1, 1).iloc[::10].index.values),
    ls='-.')
rel = 80
corner_table = []
dfrel = cdf[np.isclose(df[o.o_max_rel2c_lab], rel)]
for m in df.index.levels[0]:
    for obj in ocols_hv[1:]:
        corner_table.append(dfrel.loc[m].sort_values(by=obj,
                                                     ascending=True).iloc[0])

g = sb.PairGrid(
    df, hue='Control', hue_kws={"cmap": ["Blues", "Greens", "Reds"]})
g = g.map_diag(sb.kdeplot, lw=3)
g = g.map_offdiag(sb.kdeplot, lw=1)
