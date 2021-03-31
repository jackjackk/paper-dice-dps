from dicedps.plot.common import *


def plot_pairs_objectives_3x3():
    olist = [obj[4:] for obj in o.oset2labs['greg4']]
    flist = indata('bymiu_*.ref')
    miulist = list(map(filename2miu, flist))
    df = pd.concat([pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist], keys=miulist,
                   names=['Control', 'Idsol'])

    # df = df.loc[].reset_index().drop('Idsol', axis=1)
    df
    fig, axs = plt.subplots(3, 3, figsize=(8, 7))
    cols = np.array(sb.color_palette())[[0, 3]]
    # prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
    for i1 in range(3):
        obj1 = olist[i1 + 1]
        for i2 in range(0, i1 + 1):
            obj2 = olist[i2]
            ax = axs[i1, i2]
            prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
            for miu, p in zip([cdpstdt4, copen], cols):
                ax.scatter(df.loc[miu, obj2], df.loc[miu, obj1], alpha=1, s=2, color=p)
            ax.set_ylabel(obj1)
            ax.set_xlabel(obj2)
    for ax in axs[np.triu_indices(3, 1)]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(inplot('fig_obj_pairs.png'), dpi=200)


def plot_3d_objectives_vs_greg_2x3panel():
    olist = [obj[4:] for obj in o.oset2labs['greg4']]
    flist = indata('bymiu_*.ref')
    miulist = list(map(filename2miu, flist))
    dfboth = pd.concat([pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist], keys=miulist, names=['Control','Idsol']).loc[[copen,cdpstdt4]]
    s=scaler.fit(dfboth[['Expected_Utility_BGE']])
    fig = plt.figure(figsize=(8,4))
    for j, ccurr in zip(range(2), [copen,cdpstdt4]):
        df = dfboth.loc[ccurr]
        x = df['Expected_NPV_Mitigation_Cost'].values
        y = df['Expected_NPV_Damage_Cost'].values
        z = -df['Reliability_2C'].values
        c = s.transform(df[['Expected_Utility_BGE']])[:, 0]
        for i, ang in zip(range(1,4), [-40,-80,-120]):
            ax = fig.add_subplot(2,3,i+j*3, projection='3d')
            cmap=mpl.cm.get_cmap('winter')
            ax.scatter(xs=x,ys=y,zs=z,color=cmap(c))
            plt.axis('on')
            ax.scatter(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[1], s=40, color='r')
            ax.set_xlabel('MitCost')
            ax.set_ylabel('DamCost')
            #ax.set_zlabel('Rel2C')
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
            #ax.set_zticklabels([])
            #ax.set_zlim([0,20])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            #[t.set_visible(False) for t in ax.xaxis.get_major_ticks()]
            #[t.set_visible(False) for t in ax.yaxis.get_major_ticks()]
            #[t.set_visible(False) for t in ax.zaxis.get_major_ticks()]
            ax.view_init(elev=20, azim=ang)
    fig.tight_layout()
    fig.savefig(inplot('fig_dinosaur.png'),dpi=200)


def plot_parallel_plot_ranges_for_given_reliability():
    olist = [obj[4:] for obj in o.oset2labs['greg4']]
    flist = indata('bymiu_*.ref')
    miulist = list(map(filename2miu, flist))
    df = pd.concat([pd.read_csv(f, names=olist, sep=' ', header=None) for f in flist], keys=miulist,
                   names=['Control', 'Idsol']).loc[[copen, cdpstdt4]]
    # df[(df['Reliability_2C']<-66) & (df['Expected_NPV_Mitigation_Cost']<2.9)].groupby('Control').count()
    ndf = df.copy()
    scaled_values = scaler.fit_transform(df)
    ndf.loc[:, :] = scaled_values
    ndf['Reliability_2C'] = ndf['Reliability_2C'].round(2).mul(100).astype(int)
    ndf.groupby('Control')['Reliability_2C'].describe()

    ndfrel = ndf.groupby(['Control', 'Reliability_2C']).mean().unstack('Control').reindex(range(1, 101)).interpolate(
        axis=0)  # .stack('Control').reorder_levels([1,0]).sort_index()
    ndfrel_min = ndf.groupby(['Control', 'Reliability_2C']).min().unstack('Control').reindex(range(1, 101)).interpolate(
        axis=0)  # .stack('Control').reorder_levels([1,0]).sort_index()
    ndfrel_max = ndf.groupby(['Control', 'Reliability_2C']).max().unstack('Control').reindex(range(1, 101)).interpolate(
        axis=0)  # .stack('Control').reorder_levels([1,0]).sort_index()
    ndfrel.index = ndfrel.index.astype(float) / 100
    ndfrel_min.index = ndfrel_min.index.astype(float) / 100
    ndfrel_max.index = ndfrel_max.index.astype(float) / 100
    ndfrel.xs(copen, 1, 1)
    clp()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    cmap = mpl.cm.get_cmap('cool')
    for ax, miu2fill in zip(axs, [copen, cdpstdt4]):
        for x in [0.01, 0.25, 0.5, 0.75, 1.]:
            col = cmap(x)
            hopen = ax.plot(ndfrel.xs(copen, 1, 1).loc[[x]].reset_index().values.T, color=col, ls='--', lw=2)
            hdps = ax.plot(ndfrel.xs(cdpstdt4, 1, 1).loc[[x]].reset_index().values.T, color=col, ls='-', lw=2)
            hfill = ax.fill_between(range(4), ndfrel_min.xs(miu2fill, 1, 1).loc[[x]].reset_index().values[0],
                                    ndfrel_max.xs(miu2fill, 1, 1).loc[[x]].reset_index().values[0], color=col,
                                    alpha=0.2, lw=2)
        ax.legend([hopen[0], hdps[0], hfill], ['Open loop', 'DPS(T,dT|4)', miu2fill])
        ax.set_ylabel('Normalized value')
        ax.set_xticks(range(4))
        ax.set_xticklabels(ndfrel.xs(copen, 1, 1).reset_index().columns.values)
    fig.autofmt_xdate(rotation=10, ha='right')
    fig.tight_layout()
    fig.savefig(inplot('fig_parallel_plot.png'), dpi=200)
