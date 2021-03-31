from dicedps.plot.common import *

df = load_data(ldfthinned_mitcost)

#df: pd.DataFrame = pd.read_parquet(inoutput('dicedps', f'{ldfthinned_2degyear_dps}.dat'))

simtime = get_sim2plot(mtime)
simdps = get_sim2plot(mdps, miustep=1)

lbgemit = 'bgemit'
lbgedam = 'bgedam'
lbgeutil = 'bgeutil'


#%%

bforce = False

try:
    a = load_data(ldf_fig3_tseries)
    assert (not bforce)
except:
    sclimcalib = df.index.levels[2]
    sdamfunc = df.index.levels[3]
    try:
        dict_sim = joblib.load(inoutput('dicedps', 'dict_sim.dat'))
    except:
        dict_sim = {}
    df2scat = {}
    for scen_cli in tqdm(sclimcalib):
        for scen_df in tqdm(sdamfunc):
            for l in miulist:
                k = (scen_cli, scen_df, l)
                dict_sim[k] = dice_cli(l, 100, scen_cli, f"--damfunc={scen_df}")
                s = dict_sim[k]
                for sol in tqdm(df.index.levels[1][::4]):
                    s.run(v.get_x(df.loc[l].loc[sol].loc[scen_cli].loc[scen_df]))
                    m = s._mlist[1]
                    ydict = {}
                    xx = s.C.copy()
                    for i in range(xx.shape[0]):
                        xx.iloc[i,:] = -100. * (pow((Dice.welfare(m, m.C)) / (Dice.welfare_ref(m)), 1 / (1 - m.elasmu)) - 1)
                    ydict[lbgeutil] = xx
                    xx = s.C.copy()
                    for i in range(xx.shape[0]):
                        xx.iloc[i, :] = -100. * (
                                    pow((Dice.welfare(m, (m.YGROSS - m.ABATECOST) * ((1 - m.S)[:,np.newaxis]))) / (Dice.welfare_ref(m)), 1 / (1 - m.elasmu)) - 1)
                    ydict[lbgemit] = xx
                    xx = s.C.copy()
                    for i in range(xx.shape[0]):
                        xx.iloc[i, :] = -100. * (
                                    pow((Dice.welfare(m, m.YGROSS*(1 - m.DAMFRAC) * ((1 - m.S)[:,np.newaxis]))) / (Dice.welfare_ref(m)), 1 / (1 - m.elasmu)) - 1)
                    ydict[lbgedam] = xx
                    ydict[lytemp] = s.TATM.loc[:2200].round(3) #.shift(1)
                    ydict[lygross] = s.YGROSS.loc[:2200]
                    ydict[lyfinal] = s.Y.loc[:2200]
                    ydict[lydamcost] = s.DAMFRAC.loc[:2200] * 100.
                    ydict[lyabatcost] = (s.ABATECOST.loc[:2200] / ydict[lygross]) * 100.
                    ydict[lyloss] = (1 - ydict[lyfinal] / ydict[lygross]) * 100.
                    ydict[lymiu] = s.MIU.loc[:2200].round(2).mul(100)
                    ydict[lytempdiff] = ydict[lytemp].diff().round(3)
                    df2scat[(scen_cli,scen_df,l,sol)] = pd.concat(ydict, axis=1).stack()
        #             break
        #         break
        #     break
        # break
    joblib.dump(dict_sim, inoutput('dicedps', 'dict_sim.dat'))

    a: pd.DataFrame = (pd.concat(df2scat)
                       .rename_axis([lscencli_lab, ldamfunc_lab, labatmiu, lsol_id, lyear, lsow], axis=0)
                       .rename(columns=y2lab)
                       .rename(index=cs2lab, level=lscencli_lab)
                       .rename(index=miu2lab, level=labatmiu)
                       )
    save_data(a, ldf_fig3_tseries)

a = a.rename_axis([lscencli_lab, ldamfunc_lab, labatmiu, lsol_id, lyear, lsow], axis=0)
b = (a
     .xs('1', 0, ldamfunc_lab)
     .reset_index())

#%%

#sb.lineplot(x='t', y=lyabatcost, hue=lsol_id, data=b[b[lscencli_id]=='med'])

sb.set_context('paper')

clp()
fg = sb.FacetGrid(row=lsol_id,
                  row_order=a.index.levels[3][[0,1,3,4]],
                  col=lscencli_lab,
                  col_order=[cs2lab[x] for x in ['high','med','low']],
                  hue=labatmiu,
                  hue_order=[miu2lab[x] for x in [mtime, mdps]],
                  palette=[p['color'] for p in [prop_list[1], prop_list[0]]],
                  data=b,
                  size=2,
                  sharey='row',
                  margin_titles=True)
fg.map(sb.lineplot, lyear, lytemp_lab, ci='sd')

fig: plt.Figure = fg.fig
fg.axes[0,2].legend()
#fig.legend(hleg, lleg, loc='upper center', ncol=2)

fg.fig.tight_layout()
fg.fig.savefig(inplot('fig_supp_temp_fig3.pdf'))


#%%

sb.set_context('paper')

clp()

labtemp = y2lab[lytemp]
c = a.xs('1', 0, ldamfunc_lab)
#d: pd.Series = (c[labtemp]-3).apply(lambda x: max(0, x))
ylmit = y2lab.get(lyabatcost, lyabatcost)
yldam = y2lab.get(lydamcost, lydamcost)

d = {}
l2lab = {
    lbgeutil: 'Utility',
    lbgemit: 'Mitigation costs',
    lbgedam: 'Damage costs'
}

for l in [lbgeutil, lbgedam, lbgemit]:
    d[l2lab[l]] = ((c[l] - 1)
            .xs(2100, 0, 'Year')
            .apply(lambda x: max(0, x)==0)
            .groupby(level=[0,1,2]).sum())
# d['temp'] = ((c[labtemp] - 3)
#             .xs(2100, 0, 'Year')
#             .apply(lambda x: max(0, x)==0)
#             .groupby(level=[0,1,2]).sum())
f = pd.concat(d, names=['Variable'])
g = f #.groupby(level=[0,1,2]).sum()
g.name = '% SOWs | BGE <= 1%'
fg = sb.FacetGrid(col=lscencli_lab,
                  col_order=[cs2lab[x] for x in ['high','med','low']],
                  row='Variable',
                  data=g.reset_index(),
                  margin_titles=True,
                  size=2,
                  sharey='row')
fg.map_dataframe(sb.barplot, lsol_id, g.name, hue=labatmiu, hue_order=[miu2lab[x] for x in [mtime, mdps]], palette=[p['color'] for p in [prop_list[1], prop_list[0]]])
fg.axes[0,0].legend()
fg.fig.tight_layout()
fg.fig.savefig(inplot('fig_supp_econ_fig3.pdf'))


#%%
# for miu in miulist:
#     y = df.loc[miu].xs('med',0,'climcalib').xs('1',0,'damfunc')
#     for isol, sol in tqdm(y.iterrows(), total=y.shape[0]):
#         print(isol)
#


