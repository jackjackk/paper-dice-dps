from dicedps.plot.common import *

data = load_data(ldfthinned_mitcost)

df = data[ldfthinned_mitcost]
df = df[df[o.o_min_mean2degyears_lab]<100]
#df: pd.DataFrame = pd.read_parquet(inoutput('dicedps', f'{ldfthinned_2degyear_dps}.dat'))

simtime = get_sim2plot(mtime)


simdps = get_sim2plot(mdps, miustep=1)
#simdps2 = get_sim2plot(mdps, miustep=2)
#simdps3 = get_sim2plot(mdps, miustep=3)



#%%

y: pd.DataFrame = df.xs('med',0,'climcalib').xs('1',0,'damfunc')
ydps: pd.DataFrame = y.loc[mdps]

dict_timestep = {}

dict_mius = {}

for tstep in tqdm([1, 2, 3]):
    for isol, sol in tqdm(ydps.iterrows(), total=ydps.shape[0]):
        simdps.dc._mlist[0].miu_update_tstep = tstep
        simdps.dc._mlist[0].miu_update_method = r.MiuRBFController.MUM_LIN
        dkey = (f'{mdps}_t{tstep}',isol)
        dict_timestep[dkey] = simdps.dc.run_and_ret_objs(v.get_x(sol))
        dict_mius[dkey] = simdps.get('MIU')

for isol, sol in tqdm(y.loc[mtime].iterrows(), total=y.loc[mtime].shape[0]):
    dkey = (mtime,isol)
    dict_timestep[dkey] = simtime.dc.run_and_ret_objs(v.get_x(sol))
    dict_mius[dkey] = simtime.get('MIU')

t2co: pd.Series = simdps.get('t2co')

dftimestep = pd.DataFrame(dict_timestep).T.rename(o.obj2lab,axis=1)
dfmiu = (pd.concat(
    {x: y.rename(t2co.to_dict(), axis=1)
     for x,y in dict_mius.items()})
         .stack()
         .rename_axis(['miu','isol','t','sow'], axis=0))
         # .reset_index()
         # .set_axis(['miu','isol','t','sow','val'],
         #           axis=1,
         #           inplace=False))

#yall = pd.concat([y, dftimestep])

#%% mius plot

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
         .rename_axis([labatmiu,lyear,lsol_id,'ECS'], axis=0)
         .rename(labat)
         .mul(100.)
         .reset_index()
         .replace({labatmiu: miu2lab})
         .query('Year <= 2100'))

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
savefig4paper(fg.fig, 'supp_timestep_miu')
#plt.grid()

#%%

fig: plt.Figure = None
ax: plt.Axes = None

sb.set_context('paper')
fig, ax = plt.subplots(1, 1, figsize=(w2col, hhalf))
for miu, p in zip(dftimestep.index.levels[0], prop_list):
    yy = dftimestep.loc[miu]
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
savefig4paper(fig, 'supp_timestep_pareto')

#%%

plot_objective_pairs(
    df.xs('med',0,'climcalib').xs('1',0,'damfunc'),
    orows=[o.o_min_cbgemitcost_lab],
    ocols=[o.o_min_mean2degyears_lab],
#    axs=ax_pareto,
)

#%%
asol = df.sort_values(by=o.o_min_mean2degyears_lab).iloc[[100]]

simdps.dc._mlist[0].miu_update_tstep = 3
simdps.dc._mlist[0].miu_update_method = r.MiuRBFController.MUM_LIN
simdps.dc.run_and_ret_objs(v.get_x(asol.iloc[0]))
plot_var_cmap(simdps,asol)

