from dicedps.plot.common import *

data = load_data(ldfthinned_mitcost)

df = data[ldfthinned_mitcost]
df = df[df[o.o_min_mean2degyears_lab]<100]
#df: pd.DataFrame = pd.read_parquet(inoutput('dicedps', f'{ldfthinned_2degyear_dps}.dat'))

ccsyrlist = [2150,2050]
sccsyrlist = list(map(str, ccsyrlist))

try:
    sims = load_data(lsims_ccs)[lsims_ccs]
except:
    sims = defaultdict(dict)
    for m in [mtime,mdps]:
        for c in ccsyrlist:
            dk = dict(setup=dict(yearccs=c))
            sims[m][str(c)] = get_sim2plot(miu=m, dice_kwargs=dk)

    save_data(sims, lsims_ccs)

#%%

try:
    raise Exception()
    data = load_data([ldfccs_miu, ldfccs])
    dfccs = data[ldfccs]
    dfmiu = data[ldfccs_miu]
except:
    y: pd.DataFrame = df.xs('med',0,'climcalib').xs('1',0,'damfunc')

    # identify tstep > 100%

    dict_ccs = {}

    dict_mius = {}

    for m in [mtime,mdps]:
        for c in tqdm(sccsyrlist):
            for isol, sol in tqdm(y.loc[m].iterrows(), total=y.loc[m].shape[0]):
                dkey = (f'{m}_ccs{c}',isol)
                if (m == mtime) and (c == '2050'):
                    amiu = dict_mius[(f'{m}_ccs2150',isol)].iloc[:,0]

                    a_when_ccs_start = amiu[amiu > 1.].index[0]
                    a_when_zero_emi = amiu[amiu == 1.].index[0]
                    a_when_ccs_start_new = a_when_zero_emi + 5

                    a_ccs_profile = amiu.loc[a_when_ccs_start:2250].copy()
                    amiunew = amiu.copy()
                    amiunew.loc[a_when_ccs_start_new:(a_when_ccs_start_new+(2250-a_when_ccs_start))] = a_ccs_profile.values[:]
                    amiunew.loc[(a_when_ccs_start_new + (2250 - a_when_ccs_start) + 5):] = 1.
                    x = amiunew.values[1:]
                else:
                    x = v.get_x(sol)
                dict_ccs[dkey] = sims[m][c].dc.run_and_ret_objs(x)
                dict_mius[dkey] = pd.DataFrame(np.array(sims[m][c].dc._mlist[1].MIU[1:,:]), index=range(2015,2255,5))

    dfccs = pd.DataFrame(dict_ccs).T.rename(o.obj2lab,axis=1)
    save_data(dfccs, ldfccs)

    t2co: pd.Series = sims[m][c].get('t2co')

    dfmiu = (pd.concat(
        {x: y.rename(t2co.to_dict(), axis=1)
         for x,y in dict_mius.items()})
             .stack()
             .rename_axis(['miu','isol','t','sow'], axis=0))
    save_data(dfmiu, ldfccs_miu)

#%%


fig: plt.Figure = None
ax: plt.Axes = None

sb.set_context('paper')
fig, ax = plt.subplots(1, 1, figsize=(w2col, hhalf))
for miu, p in zip(dfccs.index.levels[0], prop_list):
    yy = dfccs.loc[miu]
    ax.plot(yy[o.o_min_mean2degyears_lab],
            yy[o.o_min_cbgemitcost_lab],
            '-o',
            label=miu2lab[miu], lw=1.5, **p)
    # ax.scatter(yy[o.o_min_mean2degyears_lab],
    #         yy[o.o_min_cbgemitcost_lab],
    #         **p)


ax.set_ylabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax.set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
ax.legend()
sb.despine(fig)
fig.tight_layout()
savefig4paper(fig, 'supp_ccs_pareto')


#%% mius plot

labat = 'Abatement (%)'
labatmiu = 'Abatement strategy'
ssow = dfmiu.reset_index()['sow']
t2co_quant = lambda x: ssow[ssow.sub(t2co.quantile(x)).abs().idxmin()]
yline = (dfmiu
         .unstack('isol')
         .loc[:, [1.39]]
         .stack()
         .unstack('sow')
         .loc[:, [t2co_quant(x) for x in [0.05, 0.50, 0.95]]]
         .rename(lambda x: f'{x:.1f}', axis=1)
         .stack()
         .rename_axis([labatmiu,'Year','Solution #','ECS'], axis=0)
         .rename(labat)
         .mul(100.)
         .reset_index()
         .replace({labatmiu: miu2lab})
         .query('Year <= 2200'))

#%%

sb.set_context('paper')

fg = sb.FacetGrid(data=yline,
                  col='ECS', margin_titles=True,
                  height=(hhalf*1.5/3.),
                  aspect=(w2col*1.5/3./hhalf*1.5))
fg.map_dataframe(sb.lineplot, x='Year', y=labat,
                 hue=labatmiu)
                 #style='Abatement strategy'
plt.legend()
fg.set_ylabels(labat)
fg.fig.tight_layout()
savefig4paper(fg.fig, 'supp_ccs_miu')
#plt.grid()



#
# fg = sb.FacetGrid(hue='miulab', data=df)
# fg.map(sb.distplot, o.o_min_q95damcost_lab)
# fg.add_legend()
#
# ares = {}
# bres = {}
# for m in [mtime,mdps]:
#     for c in ['ccs', 'noccs']:
#         for i in isols:
#             ares[(m,c,i)] = sims[m][c].dc.run_and_ret_objs(v.get_x(df.loc[m].loc[i]))
#             if (c,i) in [('ccs', 0.9), ('noccs', 1.09)]:
#                 bres[(m,c)] = sims[m][c].get('TATM')
#
# dfres = pd.DataFrame(ares).T
# dfres[np.isclose(dfres[o.o_min_cbgemitcost], 1., atol=1e-1)][[o.o_min_cbgemitcost,o.o_min_mean2degyears]]
#
#
# fg = sb.FacetGrid(col='miu', hue='c', data=pd.concat(bres, names=['miu','c']).xs(2100, 0, 2).stack().reset_index())
# fg.map(sb.distplot, 0)
# fg.add_legend()
#
#
# dftemp = pd.DataFrame(bres)
# pd.concat({'ccs':simtime_ccs.get('MIU').mean(1),
#            'noccs':simtime_noccs.get('MIU').mean(1)}).unstack(0).plot()
#
# dfmiu = pd.concat({'ccs':simdps_ccs.get('MIU').stack(),
#            'noccs':simdps_noccs.get('MIU').stack()}, names=['x','t','sow']).reset_index()
#
#
#
#
# sb.tsplot(data=dfmiu, time='t', unit='sow', value=0, condition='x', err_style='unit_traces', n_boot=0)
#
#
