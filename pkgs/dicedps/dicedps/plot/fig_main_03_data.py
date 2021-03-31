import string
from collections import defaultdict

from dicedps.plot.common import *
from paradice.dice import Damages

df: pd.DataFrame = load_rerun()

dfobj = df[o.oset2labs[o.oset_v3]]



simtime = get_sim2plot(mtime)
simdps = get_sim2plot(mdps)

#ycol = o.o_min_loss_util_bge_lab

ocol = f3xcol
#ocol = o.o_min_cbgedamcost_lab
tcol = 'thin_'+ocol
tcolmin, tcolmax = df[ocol].round(2).describe()[['min', 'max']]
zcol = f3ycol

number_of_policies = 20
dictdfthinned = {}
for miu in [mdps, mtime]:
    dfcurr= df.loc[miu]
    dfnom = dfcurr.loc['med'].loc['1']
    dfbyid = dfcurr[zcol].unstack().T
    for x in tqdm(np.nditer(np.linspace(tcolmin, tcolmax, number_of_policies))):
        idx = dfnom[np.isclose(dfnom[ocol], x, atol=1e-3)].index
        idxbest = dfbyid.loc[idx].mean(axis=1).sort_values().index[0]
        dictdfthinned[(miu,round(float(x), 2))] = dfcurr.xs(idxbest, 0, 'idsol')


dfthinned_mitcost = (
    pd.concat(dictdfthinned, names=['miulab', tcol, 'climcalib', 'damfunc']).sort_index())

save_data(dfthinned_mitcost, ldfthinned_mitcost)


# a = dfthinned_mitcost.xs('low', 0, 'climcalib').xs('1', 0, 'damfunc').xs(0.3, 0, tcol)
# simtime2 = get_sim2plot(mtime, nsow=1000, cli='low', damfunc='1', obj='v3')
# simdps2.dc.run_and_ret_objs(v.get_x(a.loc[mdps]))
# simtime2.dc.run_and_ret_objs(v.get_x(a.loc[mtime]))
# plot_var_cmap(simdps2, a, yy=["MIU", "DAMAGES"])
# plot_var_cmap(simtime2, a, yy=["MIU", "DAMAGES"])
# simtime2.get('DAMAGES')
# simdps2.get('t2co').describe()
# plot_miu_cmap(simdps, a)

# dfthinned_q95 = get_thinned_paretos(dfobj,
#                                         thin_cols=o.o_min_q95damcost_lab,
#                                     thin_sort_kws=my_sort_pareto_kws,
#                                     nroll=None,
#                                         thin_muls=20)

# # region supp figure showing pairs plot
# dfcurr = dfobj.xs('med', 0, 'climcalib').xs('1', 0, 'damfunc')
# plot_objective_pairs(
#     dfcurr,
#     orows=[o.o_min_q95damcost_lab],
#     ocols=[o.o_min_cbgemitcost_lab])
#
#     dfthinned_q95
# #dfthinned_mitcost
# df.head()
# # endregion


# # region supp figure showing VOI
# fig, axs = plt.subplots(2,1)
# i = 0
# for scen_cli in dfthinned_mitcost.index.get_level_values('climcalib').unique():
#     dfcli = dfthinned_mitcost.xs(scen_cli, 0, 'climcalib')
#     for scen_df in dfthinned_mitcost.index.get_level_values('damfunc').unique():
#         dfclidf = dfcli.xs(scen_df, 0, 'damfunc')
#         p = prop_list[i]
#         i = i + 1
#         for miu, ls in zip([mdps, mtime], ['-', '--']):
#             x = dfclidf.xs(miu, 0, 'miulab').index/100
#             y = dfclidf.xs(miu, 0, 'miulab')[o.o_min_q95damcost_lab].values
#             axs[0].plot(x, y, ls=ls, **p)
#         axs[1].annotate(f'{scen_cli}{scen_df}', xy=(x[20], y[20]))
#         diff_mitcost = dfclidf[o.o_min_q95damcost_lab].unstack('miulab').diff(axis=1)[mtime]
#         x = diff_mitcost.index/100
#         y = diff_mitcost.values
#         axs[1].plot(x, y, **p)
#         axs[1].annotate(f'{scen_cli}{scen_df}', xy=(x[20], y[20]))
# # endregion


# main figure
#write_macro = write_macro_generator('main_03')



#mpl.rcParams.update(mpl_nature)

#%% recalc regret distribution

dfthinned_mitcost.index.levels[3]

dict_rerun = {}
dict_sim = {}

#%% calc
for scen_cli in tqdm(dfthinned_mitcost.index.levels[2]):
    for scen_df in tqdm(dfthinned_mitcost.index.levels[3]):
        for l in [mtime,mdps]:
            k = (scen_cli, scen_df, l)
            if not k in dict_sim:
                dict_sim[k] = dice_cli(l, 100, scen_cli, f"--damfunc={scen_df}")
            s = dict_sim[k]
            for sol in tqdm(dfthinned_mitcost.index.levels[1][::4]):
                s.run(v.get_x(dfthinned_mitcost.loc[l].loc[sol].loc[scen_cli].loc[scen_df]))
                m = s._mlist[1]
                wref = Dice.welfare_ref(m)
                w = Dice.welfare(m, m.YGROSS*(1 - m.DAMFRAC) * ((1 - m.S)[:,np.newaxis]))
                dict_rerun[(scen_cli, scen_df,l, sol, 'damcost')] = -100.*(pow(w/wref, 1/(1 - m.elasmu)) - 1)
                w = Dice.welfare(m, (m.YGROSS - m.ABATECOST) * ((1 - m.S)[:, np.newaxis]))
                dict_rerun[(scen_cli, scen_df, l, sol, 'mitcost')] = -100. * (pow(w / wref, 1 / (1 - m.elasmu)) - 1)
                dict_rerun[(scen_cli, scen_df, l, sol, '2degyr')] = np.sum((np.maximum(0, m.TATM[1:-10]-2))*m.tstep, axis=0)
                dict_rerun[(scen_cli, scen_df, l, sol, 'maxtemp')] = np.max(m.TATM[1:-10],axis=0)
                dict_rerun[(scen_cli, scen_df, l, sol, 'miu2030')] = m.MIU[4]
                dict_rerun[(scen_cli, scen_df, l, sol, 'tatm2025')] = m.TATM[3]
                dict_rerun[(scen_cli, scen_df, l, sol, 'miu2050')] = m.MIU[8]
                dict_rerun[(scen_cli, scen_df, l, sol, 'tatm2050')] = m.TATM[7]
                dict_rerun[(scen_cli, scen_df, l, sol, 'damcost2100')] = m.DAMFRAC[18]
#joblib.dump(dict_sim, inoutput('dicedps', 'dict_sim.dat'))
joblib.dump(dict_rerun, inoutput('dicedps', 'dict_rerun.dat'))

#%%

dict_sim = joblib.load(inoutput('dicedps', 'dict_sim.dat'))
dict_rerun = joblib.load(inoutput('dicedps', 'dict_rerun.dat'))

#%%
y = pd.DataFrame(dict_rerun).T.stack().unstack(2)
y2 = y.stack().reset_index().set_axis(['cli','df','sol','var','sow','miu','val'],1,inplace=False)
dy = y[mdps] - y[mtime]
yy = dy.reset_index().set_axis(['cli','df','sol','var','sow','val'],1,inplace=False)

#%%
yyy = (yy.set_index(['cli','df','sol','var','sow'])['val']
    .unstack('sow').T.describe().T.loc[:,['25%','50%','75%']].stack()
    .unstack('var').reset_index())

y3 = (yy.set_index(['cli','df','sol','var','sow'])['val']
    .unstack('sow').stack()
    .unstack('var').reset_index())

y4 = y3.query('df=="1"')
y4 = y2.query(f'var=="maxtemp"')
y4.loc[:, 'scen'] = y4.apply(lambda x: x['cli']+x['df'], axis=1)
y4.loc[:, 'val2'] = y4.apply(lambda x: x['val']<2., axis=1)
df_rel2c = y4.groupby(['cli','df','sol','miu'])['val2'].sum().unstack('miu').diff(axis=1).loc[:,'time2'].reset_index()

#%%

sb.scatterplot(x='sol',y='time2',hue='cli',data=df_rel2c)
y4
sb.lineplot(x='sol', y='val', units='sow', estimator=None, data=y4)


#%%
s = dict_sim[('med', '1', mtime)]
dft: pd.DataFrame = df.loc[mtime]
dft1dam = dft[dft[o.o_min_cbgedamcost_lab]<1.]
asol=dft1dam[dft1dam[o.o_min_cbgemitcost_lab]<np.quantile(dft1dam[o.o_min_cbgemitcost_lab],0.10)].sort_values(o.o_min_mean2degyears_lab).iloc[0]
s.run(v.get_x(asol))


sb.lineplot(x='year', y='value', err_style='band', ci='sd', data=get_variable_from_dice(s, 'DAMFRAC').stack().reset_index().set_axis(['year','sow','value'],axis=1,inplace=False))

sol_max_mitcost = dft.sort_values(o.o_min_cbgemitcost_lab, ascending=False).iloc[0]
a = dft[(dft['dv2']>0.4) & (dft['dv6']>0.6)].sort_values(o.o_min_cbgemitcost_lab).iloc[0]

#%%
sb.jointplot(x='mitcost',
           y='damcost',
           data=y4)

#%%
sb.catplot(x='mitcost',
           y='damcost',
           data=yyy,
           style='df',
           hue='cli',
           kind='scatter')
#sb.distplot(y[mdps]-y[mtime])

#%% damage function

m = simtime.dc._mlist[1]
m.TATM[0,:] = np.linspace(1,5,100).round(2)

dict_damfunc = {}
for i, l in enumerate(ldam2):
    dict_damfunc[l] = pd.Series(100*Damages.dam2func[i+1](m, 0), index=m.TATM[0])
df_damfunc = pd.DataFrame(dict_damfunc)

save_data(df_damfunc, ldf_damfunc)


#%% climate sensitivity

dict_csdist = {}
for cslev, cslab in zip(lcslevs, lcslabs):
    dfcs = u.get_sows_setup_mcmc(h.args2climcalib(cslev), nsow=1000)
    dict_csdist[cslab] = dfcs['setup']['t2co']
df_csdist = pd.DataFrame(dict_csdist)

save_data(df_csdist, ldf_csdist)

#%% trade offs

dict_interp_mitcost = {}
dict_thinned_diffs = {}
for scen_cli in lcslevs:
    for scen_df in range(1,4):
        scen = f'{scen_cli}{scen_df}'
        dfclidf_mitcost = (dfthinned_mitcost
                           .xs(scen_cli, 0, 'climcalib')
                           .xs(str(scen_df), 0, 'damfunc'))
        dict_interp_mitcost[scen] = (dfclidf_mitcost
                    .groupby(['miulab', dfclidf_mitcost[f3xcol].round(2)]).mean()
                    .sort_index()
                    .unstack('miulab')
                    .reindex(np.arange(tcolmin, tcolmax, 1e-2).round(2))
                    .interpolate()
                    .stack('miulab')
                    .swaplevel()
                    .sort_index())
        deltax = defaultdict(dict)
        deltay = defaultdict(dict)
        for idx, sol in dfclidf_mitcost.unstack('miulab').iterrows():
            deltax[sol[f3xcol][mtime] - sol[f3xcol][mdps]][''] = sol[f3ycol][mtime]
            for dcol in delta_cols:
                deltay[sol[f3xcol][mtime]][dcol] = sol[dcol][mtime] - sol[dcol][mdps]
        sb.factorplot(data=pd.DataFrame(deltay).T.stack().reset_index().set_axis(['x','var','val'],1,inplace=False),
                      x='x',y='val',col='var')
        dict_thinned_diffs[scen] = pd.concat([
            pd.Series(deltax, name='dx').sort_index().reset_index().rename(columns={'index':'idx'}),
            pd.Series(deltay, name='dy').sort_index().reset_index().rename(columns={'index':'idy'})],
            axis=1)

df_interp_mitcost = pd.concat(dict_interp_mitcost)
save_data(df_interp_mitcost, ldf_interp_mitcost)

df_thinned_diffs = pd.concat(dict_thinned_diffs)
save_data(df_thinned_diffs, ldf_thinned_diffs)


#%% rest




# ax_diff2.set_ylabel(obj2lab2[ycol])
# ax_diff2.set_xlabel('VOI')
# ax_diff2.set_ylim(ax_troff.get_ylim())
# #ax_diff2.set_xlim([0,0.5])


#ax_troff.ticklabel_format(style='plain',axis='y',useOffset=False)
#ax_troff.get_yaxis().get_major_formatter().labelOnlyBase = False


min_95damcost_pct = df.loc[scenlist][ycol].min()
max_95damcost_pct = df.loc[scenlist][ycol].max()
abau = simtime.dc.run(np.zeros(47))
cons_2015_tusd = abau.C.loc[2015].mean()
min_95damcost_tusd = cons_2015_tusd*min_95damcost_pct/100
max_95damcost_tusd = cons_2015_tusd*max_95damcost_pct/100
write_macro('fig03-min-95damcost-pct', f'{min_95damcost_pct:.1f}', True)
write_macro('fig03-min-95damcost-Tusd', f'{min_95damcost_tusd:.1f}')
write_macro('fig03-max-95damcost-pct', f'{max_95damcost_pct:.1f}')
write_macro('fig03-max-95damcost-Tusd', f'{max_95damcost_tusd:.1f}')

"""
a.loc[a.argmax()]
a.describe()
a.reset_index()
sb.factorplot(x='mitcost',y='time',hue='scen',order=np.arange(0,23,1e-1),data=a.reset_index())
"""

#### SUPP PLOT

fig = plt.figure(figsize=(w2col,1.5*hhalf))
gs = GridSpec(4, 2)
ax_troff = plt.subplot(gs[:2, :])  # a. Trade-off 95th dam cost vs mit cost
ax_x1 = [plt.subplot(gs[x,0]) for x in [2,3]]  # b-c. Abatement and damage
ax_x2 = [plt.subplot(gs[x,1]) for x in [2,3]]  # d-e.

hs = []
ydict = defaultdict(dict)
yotherdict = defaultdict(dict)
scenlist = ['low1','med2','hig3','med1']
for scen, p in zip(scenlist, prop_list2):
    for miu, ls in zip([mdps, mtime], ['-','--']):
        dfcurr = df.loc[scen].loc[miu].loc[ids2plot[miu]]
        #ax_troff.scatter(dfcurr[xcol], dfcurr[ycol], s=3, **p)
        y2plot = pd.Series(dfcurr[ycol].values,index=dfcurr[xcol].round(2)).groupby(o.o_min_cbgemitcost_lab).mean().reindex(np.arange(0,1.25,1e-2)).interpolate()
        yother = pd.Series(dfcurr[xcol].values, index=dfcurr[ycol].round(2)).groupby(
            ycol).mean().reindex(np.arange(0, 23, 1e-2)).interpolate()
        ydict[scen][miu]=y2plot
        yotherdict[scen][miu]=yother
        #sb.regplot(dfcurr[xcol], dfcurr[ycol], order=8, line_kws=dict(lw=1.5, ls=ls), scatter_kws=dict(s=3), **p)
        #hs.append(ax_troff.plot(dfcurr[xcol], dfcurr[ycol], ls=ls, lw=1.5, **p)[0])
        hs.append(ax_troff.plot(y2plot.index, y2plot.values, ls=ls, lw=1.5, **p)[0])
    ax_troff.annotate(s=stidy(scen2lab[scen]), xy=(dfcurr[xcol].iloc[-1], dfcurr[ycol].iloc[-1]), xytext=(0,scen2off.get(scen, 15)), textcoords='offset points', ha='right', va='bottom')
ax_troff.legend(hs[-2:],[miu2lab[mdps],miu2lab[mtime]])
ax_troff.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab])
ax_troff.set_ylabel(obj2lab2[ycol].replace('(','\n('))
ax_troff.set_yscale('log')
ax_troff.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
ax_troff.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.1f'))


a=pd.concat({s:pd.concat(yotherdict[s]) for s in scenlist},names=['scen','miu','mitcost']).unstack('miu').diff(axis=1).iloc[:,1]
amax = a.argmax()
b=df.loc[amax[0]].loc[mdps]
bdamcostmin=b[ycol].min()

simdps = get_sim2plot(mdps, 100, cli=amax[0][:3], damfunc=amax[0][-1])
flatpolicies_by_mitcost = b[np.isclose(b[ycol], bdamcostmin) & (b[o.o_min_cbgemitcost_lab]<=1.25)].sort_values(o.o_min_cbgemitcost_lab)
for pol_idx, axs, lab in zip([0,-1], [ax_x1, ax_x2], ['A1', 'A2']):
    pol = flatpolicies_by_mitcost.iloc[pol_idx]
    plot_var_cmap(simdps,
                  amiu=v.get_x(pol),
                  yy=['MIU','TATM'],
                  axs=axs)
    ax_troff.scatter(pol[xcol], pol[ycol], color='k')
    ax_troff.annotate(lab, xy=(pol[xcol], pol[ycol]), xytext=(5, 5), textcoords='offset pixels', va='bottom', ha='left', fontsize=8)
    axs[0].set_title(lab)

savefig4paper(fig, 'supp_expdam')



def write_datafile_for_interactive_webpage():
    my_sort_pareto_kws = dict(
        by=[
            o.o_min_cbgemitcost_lab,
            o.o_min_mean2degyears_lab,
            o.o_min_cbgedamcost_lab,
            o.o_min_loss_util_bge_lab,
        ],
        ascending=True,
    )
    dfthinned_2degyear_dps = (
        get_thinned_paretos(df.loc[mdps].loc['med'].loc['1'],
                            thin_cols=[o.o_min_cbgemitcost_lab,
                                       o.o_min_mean2degyears_lab],
                            thin_muls=[50, 1],
                            thin_sort_kws=my_sort_pareto_kws,
                            nroll=None))

    ocolset = ['obj_Mean_Degree_Years_Above_2C',
               'obj_Expected_Utility_Loss_BGE',
               'obj_Expected_CBGE_Mitigation_Cost',
               'obj_Expected_CBGE_Damage_Cost',
               'obj_Abatement_2030',
               'obj_Abatement_2050',
               'obj_Min_q95_Damage_Cost',
               'obj_Min_q95_Max_Temperature']

    (dfthinned_2degyear_dps[ocolset]
        .assign(obj_Abatement_2030=100. * dfthinned_2degyear_dps['obj_Abatement_2030'] / 2.,
                obj_Abatement_2050=100. * dfthinned_2degyear_dps['obj_Abatement_2050'] / 2.)
        .rename(dict(zip(ocolset, ['Warming (2deg-years)',
                                   'Utility loss (%CBGE)',
                                   'Mitigation cost (%CBGE)',
                                   'Damage cost (%CBGE)',
                                   'Abat(2030)/.5degC (%)',
                                   'Abat(2050)/.5degC (%)',
                                   '95th damage cost (%BGE)',
                                   '95th max temp (degC)'])), axis=1)
        .to_csv(
        '/home/jack/working/dev-website-mgiacomo/static/dicedps/demo/data/dicedps.csv',
        index=False,
        float_format='%.2f')
    )
