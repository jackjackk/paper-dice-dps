import string
from typing import Dict

from dicedps.plot.common import *

df = load_merged(orient='min')
df = df[df[o.o_min_mean2degyears_lab]<=100.]

save_data(df, ldffig2)

data = load_data(ldfthinned_2degyear_dps, lsol_opt_1obj)



dftime = df.loc[mtime]

# Simulators
simdps = get_sim2plot(mdps, 100)
simtime = get_sim2plot(mtime, 100)

# simdps10k = get_sim2plot(mdps, 10000)
# simtime10k = get_sim2plot(mtime, 10000)


#%% find points

dfquery = lambda miu, objval, obj: df.loc[miu].iloc[(df.loc[miu][obj]-objval).abs().argsort().iloc[0]]
dfquerymit = partial(dfquery, obj=o.o_min_cbgemitcost_lab)
dfquerydeg = partial(dfquery, obj=o.o_min_mean2degyears_lab)
#df09mit = pd.concat({x: dfquerymit(x, 1.5) for x in [mtime,mdps]})


df09mit = get_sol_by_query(df, {o.o_min_cbgemitcost_lab: (1.5, 1e-3)})
hpoints = {}

hpoints["A"] = data[lsol_opt_1obj].iloc[0]
hpoints["X"] = df09mit.loc[mdps]
hpoints["Y"] = df09mit.loc[mtime]
hpoints["Z"] = dfquerydeg(mtime, hpoints['X'][o.o_min_mean2degyears_lab])

save_data(hpoints, lhpoints)


#%% compute mitcosts, miu and temp for x, y, z

dfxyz = defaultdict(dict)
for s, sim in zip(["X", "Y", "Z"], [simdps10k, simtime10k, simtime10k]):
    p = hpoints[s]
    sim.dc.run(v.get_x(p))
    dfxyz[lhpoints_mitcosts][s] = 100.*sim.get('ABATECOST').div(sim.get("YGROSS"))
    if sim == simtime10k:
        dfxyz[lhpoints_miu][s] = sim.get("MIU").mean(axis=1) * 100.
    else:
        dfxyz[lhpoints_miu][s] = sim.get("MIU") * 100.
    dfxyz[lhpoints_temp][s] = sim.get("TATM")

save_data(dfxyz, lhpoints_df)


#%% compute mitcosts, miu and temp for x, y, z under deep uncertainty

#dfxyz2 = defaultdict(defaultdict(dict))
dfxyz2 = {}
simlist = defaultdict(dict)

dfdeep = load_data(ldfthinned_mitcost)
sclimcalib = dfdeep.index.levels[2]

for l in miulist:
    for scen_cli in tqdm(sclimcalib):
        simlist[l][scen_cli] = get_sim2plot(l, 1000, cli=scen_cli)


for s, smiu in zip(["X", "Y", "Z"], [mdps, mtime, mtime]):
    p = hpoints[s]
    for k, sim in simlist[smiu].items():
        sim.dc.run(v.get_x(p))
        dfxyz2[(lhpoints_mitcosts,s,k)] = 100.*sim.get('ABATECOST').div(sim.get("YGROSS"))
        if smiu == mtime:
            dfxyz2[(lhpoints_miu,s,k)] = sim.get("MIU").mean(axis=1) * 100.
        else:
            dfxyz2[(lhpoints_miu,s,k)] = sim.get("MIU") * 100.
        dfxyz2[(lhpoints_temp,s,k)] = sim.get("TATM")

save_data(pd.concat(dfxyz2, names=['v','p','cli','t']).stack(), lhpoints_dfdeep)

#%% not used

#
# """
# AN1_mitcosts_pct = [hpoints[x][o.o_min_cbgemitcost_lab] for x in ["A", "N1"]]
# AN1_mean_mitcost_pct = sum(AN1_mitcosts_pct) / 2.0
# print(f"fig02-ex-mitcost-pct: {AN1_mitcosts_pct}")
# write_macro("fig02-ex-mitcost-pct", f"{AN1_mean_mitcost_pct:.1f}", reset=True)
# AN1_mean_mitcost_tusd = cons_2015_tusd * AN1_mean_mitcost_pct / 100.0
# write_macro("fig02-ex-mitcost-Tusd", f"{AN1_mean_mitcost_tusd:.2f}")
# N1_2dy = hpoints["N1"][o.o_min_mean2degyears_lab]
# A_2dy = hpoints["A"][o.o_min_mean2degyears_lab]
# N1mA_2dy = -(A_2dy - N1_2dy)
# N1mA_2dy_pct = N1mA_2dy / N1_2dy * 100.0
# write_macro("fig02-N1-2degyears", f"{np.round(N1_2dy, -1):.0f}")
# write_macro("fig02-N1-minus-A-2degyears", f"{N1mA_2dy:.0f}")
# write_macro("fig02-N1-minus-A-2degyears-pct", f"{N1mA_2dy_pct:.0f}")
# AN2_2dys = [hpoints[x][o.o_min_mean2degyears_lab] for x in ["A", "N2"]]
# print(f"test: {AN2_2dys}")
# AN2_mean_2dy = sum(AN2_2dys) / 2.0
# write_macro("fig02-ex-2degyears", f"{np.round(AN2_mean_2dy, -1):.0f}")
# N2_mitcost_pct = hpoints["N2"][o.o_min_cbgemitcost_lab]
# A_mitcost_pct = hpoints["A"][o.o_min_cbgemitcost_lab]
# write_macro("fig02-N2-mitcost-pct", f"{N2_mitcost_pct:.1f}")
# write_macro("fig02-A-mitcost-pct", f"{A_mitcost_pct:.1f}")
# N2mA_mitcost_Tusd = cons_2015_tusd * (N2_mitcost_pct - A_mitcost_pct) / 100
# write_macro(
#     "fig02-N2-minus-A-mitcost-Busd", f"{(1000*np.round(N2mA_mitcost_Tusd,2)):.0f}"
# )
# """
#
#
#
# dftime = df.loc[mtime]
# dfdps = df.loc[mdps]
#
# # ydiff.loc[np.round(xa,0)]
#
#
#
#
#
# """
# for s, htype, pro in zip(["D", "E"],
#     [dict(histtype='stepfilled',
#         alpha=0.5),
#     dict(histtype='step', lw=1.5, alpha=1)],
#                        prop_list):
#     htype.update(pro)
#     htype['cumulative'] = True
#     sb.distplot(
#         temp2100[s].max(),
#         ax=ax_temp,
#         label=s,
#         kde=False,
#         norm_hist=True,
#         hist_kws=htype
#     )
#
# handles, labels = ax_temp.get_legend_handles_labels()
# handles = [handles[0]] + [Line2D([0], [0], lw=1.5, **prop_list[1])]
# labels = [f'Adaptive |\n{getxy("D")[1]:.1f}%CBGE', elab.replace('| ', '|\n')]
# ax_temp.legend(handles, labels, title=titleg, loc='lower right')
# ax_temp.set_xlabel("Temperature peak (°C)")
# ax_temp.set_ylabel("Cumulative\nDistribution")
#
# """
#
#
#
# #%% not used
#
# """
# import probscale
# ax_temp.clear()
# probscale.probplot(tpeak['D'], ax=ax_temp,
#                    bestfit=False,
#                    scatter_kws=dict(alpha=1, markersize=2), probax='y',
#                    **prop_list[0])
# probscale.probplot(tpeak['E'], ax=ax_temp,
#                    bestfit=False, scatter_kws=dict(alpha=0.75), probax='y')
# #ax_temp.invert_yaxis()
# ax_temp.set_ylabel('Non-Exceedance Probability')
#
# ax_temp.yaxis.set_major_locator(plt.FixedLocator([0.1,1,50,99,99.9]))
# """
#
#
# """
# temp_xlim = ax_temp.get_xlim()
#
#
# for thigh in [3,3.5]:
#     acdf3 = dftcdf.reindex([thigh], method='nearest', tolerance=1e-1).iloc[0]
#     ax_temp.axvline(thigh, color='k', ls='--', alpha=0.3, zorder=-100)
#     ax_temp.plot([thigh]*2, 1.-acdf3[['D','E']]/100., color='k', lw=2)
#     ax_temp.annotate(
#         f'{1-acdf3["E"]/100.:.2f} to {1-acdf3["D"]/100.:.2f}',
#         xy=(thigh,(1.-acdf3[['D','E']]/100.).sum()/2.),
#         xytext=(-5,0), textcoords='offset pixels',
#         ha='right',
#         va='center',
#     )
#
# hpoints['D']
# """
#
#
#
# """
# h = sb.distplot(
#     mitcosts['D'].max(),
#     ax=ax_mitcost,
#     label=s,
#     kde=False,
#     norm_hist=True,
#     hist_kws={"cumulative": True, "alpha": 0.5},
#     **prop_list[0],
# )
#
# non_adap_cost = mitcosts['F'].values.max()
# adap_highest_cost = mitcosts['D'].values.max()
#
# ax_mitcost.plot([0, non_adap_cost, non_adap_cost, adap_highest_cost], [0, 0, 1, 1], **prop_list[1])
#
# handles, labels = ax_mitcost.get_legend_handles_labels()
# handles = [handles[0]] + [Line2D([0], [0], lw=1.5, **prop_list[1])]
# labels = [f'Adaptive |\n{getxy("D")[0]:.0f}°C-yr', flab.replace('| ', '|\n')]
# ax_mitcost.legend(handles, labels, title=titleg)
# ax_mitcost.set_ylabel("Cumulative\ndistribution")
#
#
# mitcost_savings = lambda x: mitcosts['F'].values.max()-mitcosts['D'].max().quantile(x)
# ax_mitcost.annotate(
#     f'{mitcost_savings(0.5):.1}%',
#     xy=(mitcosts['F'].values.max()-mitcost_savings(0.5)/2.,0.5),
#     xytext=(0,5), textcoords='offset pixels',
#     ha='center',
#     va='bottom',
# )
# ax_mitcost.plot((mitcosts['F'].values.max()-mitcost_savings(0.5), mitcosts['F'].values.max()),
#                 [0.5]*2, color='k', lw=2)
#
# ax_mitcost.annotate(
#     f'{mitcost_savings(0.25):.1}%',
#     xy=(mitcosts['F'].values.max()-mitcost_savings(0.25)/2.,0.25),
#     xytext=(0,5), textcoords='offset pixels',
#     ha='center',
#     va='bottom',
# )
# ax_mitcost.plot((mitcosts['F'].values.max()-mitcost_savings(0.25), mitcosts['F'].values.max()),
#                 [0.25]*2, color='k', lw=2)
# ax_mitcost.axhline(0.25, color='k', ls='--', alpha=0.3, zorder=-100)
# ax_mitcost.axhline(0.5, color='k', ls='--', alpha=0.3, zorder=-100)
# """
#
#
# # not used
#
# #abau = simtime.dc.run(np.zeros(47))
# #cons_2015_tusd = abau.C.loc[2015].mean()
#
# """
# dfvoi_mitcost = (get_value_of_information(dfthinned_2degyear,
#                                          index_by=o.o_min_mean2degyears_lab,
#                                          value_col=o.o_min_cbgemitcost_lab)
#                  .mul(cons_2015_tusd * 1000 / 100))
# #dfvoi_mitcost.plot()
#
# dfvoi_2degyear = get_value_of_information(dfthinned_mitcost,
#                                          index_by=o.o_min_cbgemitcost_lab,
#                                          value_col=o.o_min_mean2degyears_lab)
# #dfvoi_2degyear.plot()
# """
#
# #write_macro = write_macro_generator("main_02")
#
# """
# xmin, xmax = 21.240221450112973, 373.49247252276695
# # Diff
# x = np.round(np.arange(xmin,xmax),0)
# ret = {}
# for xx in x:
#     y = get_sol(df, {o.o_min_mean2degyears_lab:xx, o.o_min_cbgemitcost_lab:'min'}).unstack(1)
#     if v.get_o(y).dropna(1).shape[1] == 0:
#         continue
#     ret[xx] = y
# dfdiff = pd.concat(ret)
#
# highlev = 35.8
# y = dfdiff.loc[highlev]
# A = y.loc[mdps]
# """
