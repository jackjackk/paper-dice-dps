from dicedps.plot.common import *

dfxyz = load_data(lhpoints_dfdeep)

#%%

lmiu = 'Abatement (% Base CO2)'
lmitcosts = 'Mitigation costs'
ltemp = 'Temperature (°C)'
dfxyz = (dfxyz.rename_axis(['Variable','Solution',lscencli_lab, lyear, lsow], axis=0)
     .unstack('Variable')
     .rename(columns={
    lhpoints_mitcosts: lmitcosts,
    lhpoints_miu: lmiu,
    lhpoints_temp: ltemp,
}).rename(index=cs2lab, level=lscencli_lab))


#%%

clp()

sb.set_context('paper')

fig, (ax_miu, ax_mitcost, ax_temp_prob) = \
    plt.subplots(3,1,figsize=(w2col,hhalf*1.5))


#%%

ax_miu.clear()
sb.boxplot(x='Year',
            y=lmiu,
            data=(dfxyz[lmiu]
                  .loc['X']
                  .loc[(slice(None),slice(2100),slice(None))]
                  .rename(index={x: f'X ({x})' for x in cs2lab.values()}, level=lscencli_lab)
                  .reset_index()),
            hue=lscencli_lab,
            ax=ax_miu,
whis=[5, 95], width=0.5, showfliers=False,
           medianprops={'lw':2}
            )

ax_miu.plot(dfxyz[lmiu].loc['Y'].xs('Medium', 0, lscencli_lab).xs(0,0, lsow).loc[:2100].values, color='0.2', marker='o', markerfacecolor='.7', ls='-', lw=2, label='Y')
ax_miu.plot(dfxyz[lmiu].loc['Z'].xs('Medium', 0, lscencli_lab).xs(0,0, lsow).loc[:2100].values, color='0.2', marker='s', markerfacecolor=prop_list[6]['color'], ls='-.', lw=2, label='Z')
ax_miu.legend()

#%%

ax_mitcost.clear()
mitcosts_peak = (dfxyz[lmitcosts]
                 .groupby(level=['Solution', 'Climate', 'SOW'])
                 .max()
                 .loc[['X','Z']])

#mitcosts_peak[mitcosts_peak.scen=='D'].quantile(.05)
sb.boxplot(y='Solution', x=lmitcosts, data=mitcosts_peak.reset_index(), hue=lscencli_lab, ax=ax_mitcost,
           whis=[5, 95], width=0.5, showfliers=False,
           medianprops={'lw':2})

ax_mitcost.set_yticklabels(['X','Z'])
ax_mitcost.set_ylabel(None)

ax_mitcost.set_ylim([1.5,-0.8])
ax_mitcost.set_xlabel("Mitigation cost peak\n(% Gross GDP / year)")

#%%

ax_temp_prob.clear()
tpeak = (dfxyz[ltemp].groupby(level=['Solution', 'Climate', 'SOW'])
                 .max()).unstack('SOW').T

x = np.linspace(2.5, 4, 50)
tcdf = {}
for i, y in enumerate(x):
    tcdf[y] = tpeak[tpeak>y].count()/1e3*100

dftcdf = pd.concat(tcdf, names=[ltemp, 'Solution', lscencli_lab]).unstack('Solution')
dftcdf['Difference'] = dftcdf['X']-dftcdf['Y']
lratio = 'Probability of exceeding \ngiven temperature under X\ndivided by the corresponding \nprobability under Y'
dftcdf[lratio] = dftcdf['X']/dftcdf['Y']

sb.lineplot(x=ltemp, y=lratio, data=dftcdf[lratio].reset_index(), hue=lscencli_lab, ax=ax_temp_prob)

ax_temp_prob.legend()
#ax_temp_prob.set_xlim(temp_xlim)
ax_temp_prob.set_xlabel("Temperature (°C)")
#ax_temp_prob.set_ylabel('Ratio of exceedence probabilities')

#%%

sb.despine(fig)
fig.tight_layout()

savefig4paper(fig, 'supp_xyz_deep')