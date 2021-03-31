from dicedps.plot.common import *

df = load_merged(orient='min')


#%% Thinned Pareto
my_sort_pareto_kws = dict(
            by=[
                o.o_min_cbgemitcost_lab,
                o.o_min_mean2degyears_lab,
                o.o_min_cbgedamcost_lab,
                o.o_min_loss_util_bge_lab,
            ],
            ascending=True,
        )

dfthinned_2degyear = (
    get_thinned_paretos(df,
                        thin_cols=[o.o_min_cbgemitcost_lab,
                                   o.o_min_mean2degyears_lab],
                        thin_muls=[100,1],
                        thin_sort_kws=my_sort_pareto_kws,
                        nroll=None))

dfthinned_2degyear[o.o_max_inv_loss_util_bge_lab] = -dfthinned_2degyear[o.o_min_loss_util_bge_lab]

dfthinned_2degyear_dps: pd.DataFrame = (
    dfthinned_2degyear.xs(mdps, 0, 'miulab')
    .reset_index(drop=True))

save_data(dfthinned_2degyear_dps, ldfthinned_2degyear_dps)


#%% Best solutions along individual objectives
dfdps : pd.DataFrame = df.xs(mdps, 0, 'miulab')
dps_best = []
for obj in ocolset:
    dps_best.append(dfthinned_2degyear_dps.sort_values(obj).iloc[[0]])
sol_opt_1obj: pd.DataFrame = pd.concat(dps_best).drop_duplicates()

save_data(sol_opt_1obj, lsol_opt_1obj)


#%% Understand extremes of the Paretos

ocols = ocolset
dfthinned_2degyear_dps[ocols].head()
dfdps_sorted = dfdps[ocols].sort_values(**default_sort_pareto_kws)
dfdps_sorted.tail()
xmax, xmin = dfdps_sorted.iloc[[0,-1],0]
myround = lambda x: round(x * 2) / 2
np.arange(myround(xmin), myround(xmax), 1/2)


#%% Reevaluate best single-obj solutions

simdps = get_sim2plot(mdps, 200)


# (dfthinned_2degyear_dps[ocolset]
#  .rename(dict(zip(ocolset,['Warming (2deg-years)',
#                            'Utility loss (%CBGE)',
#                            'Mitigation cost (%CBGE)',
#                            'Damage cost (%CBGE)'])),axis=1)
#  .to_csv(
#     '/home/jack/working/website-dicedps/demo/data/dicedps.csv',
#     index=False,
#     float_format='%.3f')
# )


# for Cooke
"""
df_all = []
mynpv = lambda x: np.npv(0.05, x)
for i, (idsol, sol) in enumerate(dfthinned_2degyear_dps.iloc[::3].iterrows()):
    simdps.dc.run(v.get_x(sol))
    ytemp_orig = simdps.get('TATM')
    ytemp = ytemp_orig.round(3).shift(1)
    ygross = simdps.get('YGROSS')
    yfinal = simdps.get('Y')
    ydamcost = simdps.get('DAMFRAC')*100.
    yabatcost_orig = simdps.get('ABATECOST')
    yabatcost = (simdps.get('ABATECOST')/ygross)*100.
    yloss = (1-yfinal/ygross)*100.

    ytemp_interp = ytemp_orig.reindex(range(2015,2201)).interpolate()
    df_metrics = []
    df_metrics.append(ytemp_interp[ytemp_interp>2].sub(2).sum())
    df_metrics.append(100.*(1.-np.apply_along_axis(mynpv, 0, yfinal)/np.apply_along_axis(mynpv, 0, ygross)))
    df_metrics.append(100.*np.apply_along_axis(mynpv, 0, yabatcost_orig)/np.apply_along_axis(mynpv, 0, ygross))
    df_metrics.append(100.*np.apply_along_axis(mynpv, 0, ydamcost/100.*ygross)/np.apply_along_axis(mynpv, 0, ygross))
    df_all.append(pd.DataFrame(np.c_[df_metrics].T, columns=['Two-degree years', 'Utility', 'Mitigation cost', 'Damage cost']))

df2csv = pd.concat(df_all, axis=0, names=['id_sol', 'sow'], keys=range(dfthinned_2degyear_dps.iloc[::3].shape[0]))
df2csv.columns = list('TUMD')
df2csv.unstack('id_sol').swaplevel(0, 1, 1).sort_index(1).to_csv('dice_adaptive_giacomo4roger.csv', float_format='%.3f')
"""

dfsol1d_reeval = reeval_sols_for_plot_temp_miu(df=sol_opt_1obj, simdps=simdps)
save_data(dfsol1d_reeval, ldfsol1d_reeval)


#%% misc

#cax.clear()
"""
cax = plt.subplot(outer_grid[0, 1])
cb = mpl.colorbar.ColorbarBase(
    ax=cax,
    cmap=mpl.cm.Greys,
    norm=mpl.colors.Normalize(vmin=tdiff_min, vmax=tdiff_max),
    orientation="vertical"
)
cax.yaxis.set_ticks_position("right")
cb.set_label('Change in\ntemperature (°C/5yr)', rotation=-90, labelpad=25)
outer_grid.tight_layout(fig, h_pad=1.5, pad=1) #, pad=0, h_pad=0, w_pad=-1) #, h_pad=1.0, w_pad=1.0) #, rect=(0, 0, 1, 1))
#outer_grid.set_width_ratios(width_ratios=[40,1])
"""




"""
from dicedps.plot.common import *

df = load_merged(orient='min')

simtime = get_sim2plot(mtime, 100)
dice = simtime.dc._mlist[1]
miu_bau = dice._bau_miu
dice.set_bau()
simtime.dc.run(miu_bau[2:])

wref = sum(((pow(m.YGROSS[t] * (1 - m.S[t]) * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1) * m.l[t] * m.rr[t] for t in m.t[1:])

fig, ax = plt.subplots(1,1)
a = Data.load(dice._bau).YGROSS #.plot(ax=ax)
b = pd.DataFrame(np.array(dice.YGROSS)) #.plot(ax=ax)
a[40]

b.loc[40,:].describe()

dftime = df.loc[mtime]
dftime.columns.tolist()
mius = v.get_x(dftime.nsmallest(1, o.o_min_cbgemitcost_lab))
simtime.dc.run(miu)
fig, axs = plt.subplots(2,1)
for miu in mius:
    plot_var_cmap(simtime, miu, ['EIND','MIU'], axs=axs)
simtime.get('EIND')

t2150 = np.argwhere(dice.year==2150)[0,0]
miu_bau_eps = miu_bau.copy()
miu_bau_eps[t2150:] += 0.01

pd.DataFrame({'base':miu_bau,'eps':miu_bau_eps}).plot()
a = pd.Series(simtime.dc.run_and_ret_objs(miu_bau[2:]))
b = pd.Series(simtime.dc.run_and_ret_objs(miu_bau_eps[2:]))
c = (a-b).abs()
c
sum(((pow(m.YGROSS[t] * (1 - m.S[t]) * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1) * m.l[t] * m.rr[t] for t in m.t[1:])

simtime.dc._mlist[1].year
dice = simtime.dc._mlist[1]
axs[1].plot(simtime.dc._time_dice._range.year.values, dice._bau_miu[1:])
simtime.dc._mlist
"""