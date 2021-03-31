from dicedps.plot.common import *
#inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','meeting-dicedps-20180511','figures',*x)

inplot = lambda *x: join(home, 'working', 'paper-dice-dps', 'inertia', 'figures', *x)

dforig = v.load_pareto('*greg4b*_c*merged.csv', revert_max_objs=False)

#df = v.load_pareto('u1w1000doeclim_m*_i1p*_nfe4000000_objjack5_s0_seed0000_merged.csv', objlabs=False, revert_max_objs=False)
#df = v.load_pareto('u1w1000doeclim_m*_i1p*_nfe4000000_objgreg4_s1_seed0001_last.csv', objlabs=False, ndvsmap={'time':47}, revert_max_objs=False)

#df = -df[o.oset2vout['greg4']]
dfsign = -dforig[o.oset2labs['greg4b']]

# Remove "con" level
dfdict = {}
for miu, con in  [('rbfXdX44','none'),('time','inertmax')]:
    y = dfsign.loc[miu].loc[con].sort_values(o.o_min_cbgemitcost_lab)
    yy = y.groupby(y[o.o_max_rel2c_lab].round(1))
    dfdict[miu] = pd.concat([yy.first(), yy.last()], axis=0)
df = pd.concat(dfdict, names=['miu','rel2c'])
ocols_cbge = [o.o_min_cbgedamcost_lab,o.o_min_cbgemitcost_lab]
mincost = -df[ocols_cbge].abs().min().min()
maxcost = -df[ocols_cbge].abs().max().max() 
lastrecord = df.iloc[-1]
df4scaling = df.copy()
for extremecost in [mincost, maxcost]:
    for ocol in ocols_cbge:
        lastrecord[ocol] = extremecost
    df4scaling = df4scaling.append(lastrecord, ignore_index=True)
scaler.fit(df4scaling)
ndf3 = df.copy()
ndf3.loc[:, :] = scaler.transform(df)
ndf3 = ndf3.iloc[:-1,:]

rel=5
#for rel in [5, 25, 50, 58]:

rel1=33
for rel1, rel2 in zip([0,20,33],[20,33,40]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ndf3.T.plot(ax=ax, color='gray', alpha=0.1, ls='-', lw=1, legend=False)
    nrel1 = scaler.transform(np.array([[rel1,0,0,0]]))[0,0]
    nrel2 = scaler.transform(np.array([[rel2,0,0,0]]))[0,0]
    idx_within_thres = (ndf3[o.o_max_rel2c_lab] >= nrel1) & (ndf3[o.o_max_rel2c_lab] <= nrel2)
    ndf_curr = ndf3[idx_within_thres]
    for miu, p, cmap, normext in zip(df.index.levels[0],
                            [prop_list[0], prop_list[1]],
                            [plt.cm.copper, plt.cm.Blues_r],
                            [(-1,1),(0,2)]):
        y = ndf_curr.loc[miu].T
        a=y.loc[o.o_min_cbgemitcost_lab].values
        b=np.ones_like(a)
        ycost01 = scaler.transform(np.c_[b,b,-a,b])[:,2]
        norm = mpl.colors.Normalize(vmin=normext[0], vmax=normext[1]) #y.loc[o.o_max_rel2c_lab].min(), vmax=y.loc[o.o_max_rel2c_lab].max())
        y.plot(ax=ax, alpha=1, ls='-', lw=1.5, label=miu, legend=False, color=cmap(norm(ycost01))) # **p)
    ax.annotate(f'>{rel1}%',
        xy=(-0.1,scaler.transform(np.array([[rel1,-1,-1,-1]]))[0][0]),
        xycoords='data',
        xytext=(-5, 0),
        textcoords='offset points',
        horizontalalignment='right',
        verticalalignment='center',
        annotation_clip=False)
    ax.annotate(f'<{rel2}%',
        xy=(-0.1,scaler.transform(np.array([[rel2,-1,-1,-1]]))[0][0]),
        xycoords='data',
        xytext=(-5, 0),
        textcoords='offset points',
        horizontalalignment='right',
        verticalalignment='center',
        annotation_clip=False)
    ax.set_xticks(range(4))
    ax2 = ax.twiny()
    ax.set_xlim([-0.1, 3.1])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(4))
    #ax.set_xticklabels([f'{x:.1f}\n{s}' for x,s in zip(scaler.data_min_,['Reliability\n2C goal (% SOWs)','Utility\nLoss (% BAU)','Mitigation\nCost (% GDP)','Damage\nCost (% GDP)'])])
    #ax2.set_xticklabels([f'{x:.1f}' for x in scaler.data_max_])
    xtickgen = lambda vec: [f'{x:.1f}' if i==0 else f'{-x:.1f}' for i, x in enumerate(vec)]
    ax.set_xticklabels(xtickgen(scaler.data_min_))
    ax2.set_xticklabels([
        f'{s}\n{x}' for s, x in zip(['Reliability\n2C goal (% SOWs)', 'Utility\nLoss (% BGE)',
                                     'Mitigation\nCost (% BGE)', 'Damage\nCost (% BGE)'],
                                    xtickgen(scaler.data_max_))
    ])
    ax.tick_params(left=False)
    ax.tick_params(labelleft=False)
    hdps = plt.Line2D((0, 1), (0, 0), lw=2, **prop_list[1])
    hopen = plt.Line2D((0, 1), (0, 0), lw=2, **prop_list[0])
    hbest = plt.Line2D(
        (0, ), (0, ), color="white", marker='D', markerfacecolor='k')
    hthres = plt.Rectangle((0, 0), 1, 1, fill=False)
    l = ax.legend(
        [hdps, hopen, hbest], #, hthres],
        ['Adaptive', 'Non-adaptive', 'Preferred value'], # 'Thresholds'],
        bbox_to_anchor=(0, 1.3, 1, 0.2),
        loc="lower center",
        borderaxespad=0,
        ncol=4)
    for x, y in zip(range(4), [1, 1, 1, 1]):
        ax.scatter(x, y, color='k', marker='D', s=30, zorder=200)
    fig.tight_layout(rect=[0, 0, 1., 1.03])
    fig.savefig(inplot(f'parallel_plot_{rel1}rel.png'), dpi=200)





































for rel, ax in zip([0,33,50], axs):
        amin = y.min(1)
        amax = y.max(1)
        #ax.fill_between(range(len(amin.index)), amin.values, amax.values, alpha=0.7, **p)

clp()
ndf.T.head()
hopen = ax.plot(y.values, color='gray', ls='-', lw=1, alpha=0.05)

    idx_rel2c_odd = (df[orel2c].astype(int) % 2)
    idx_bymiu = {cdpstdt4: idx_rel2c_odd, copen: ~idx_rel2c_odd}
    #ndf['Reliability_2C'] = 1-ndf['Reliability_2C'] #.round(2).mul(100).astype(int)
    #ndf.groupby('Control')['Reliability_2C'].describe()
    #ndf2dec = ndf.round(2).drop_duplicates()
    #y_lowbound = scaler.transform([[50, 0, -3, 0]])[0]
    #idx_within_thres_2dec = ndf2dec.iloc[:, 0].mul(0).add(1).astype(bool)
    #for obj, low in zip(olist, y_lowbound):
    #    idx_within_thres_2dec = idx_within_thres_2dec & (ndf2dec[obj] >= low)
    #idx_rel2c_odd_2dec = (ndf2dec[orel2c].mul(100).astype(int) % 2)
    idx_bymiu = {cdpstdt4: idx_rel2c_odd, copen: ~idx_rel2c_odd}

    cmap_grays = mpl.cm.get_cmap('Greys')
    for cmap_lab, miu2fill, p in zip(['winter', 'autumn'], [cdpstdt4, copen],
                                       prop_list):
        cmap = mpl.cm.get_cmap(cmap_lab)
        for x, y in tqdm(
                ndf[(~idx_within_thres)].loc[miu2fill].iterrows()):
            #for x, y in tqdm(ndf[(~idx_within_thres)].loc[miu2fill].iterrows()):
            #col=cmap_grays((y['Reliability_2C']))
            hopen = ax.plot(y.values, color='gray', ls='-', lw=1, alpha=0.05)
        for x, y in ndf[idx_within_thres].loc[miu2fill].iterrows():
            #col=  #cmap((y['Reliability_2C']))
            hopen = ax.plot(y.values,ls='-', lw=2, alpha=0.6, **p)

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
    **prop_list[1])
ax.annotate(
    f'{nsol_dps}\nsolutions',
    xy=(2, (y_lowbound[2])),
    xycoords='data',
    xytext=(-55, 60),
    textcoords='offset points',
    fontweight='bold',
    horizontalalignment='center',
    verticalalignment='top',
    **prop_list[0])
ax.annotate(
    '<3',
    xy=(2, (y_lowbound[2])),
    xycoords='data',
    xytext=(0, -5),
    textcoords='offset points',
    horizontalalignment='center',
    verticalalignment='top')

