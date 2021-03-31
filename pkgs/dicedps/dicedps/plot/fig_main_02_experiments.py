dfround = {}
dfmean = {}
for mlab in [mtime, mdps]:
    dfround[mlab] = df.loc[mlab].round(
        {
            o.o_min_mean2degyears_lab: 1,
            o.o_min_loss_util_bge_lab: 2,
            o.o_min_cbgemitcost_lab: 2,
            o.o_min_cbgedamcost_lab: 2,
        }
    )  # .drop_duplicates()
    dfmean[mlab] = dfround[mlab].groupby(o.oset2labs[last_oset], as_index=False).mean()

# fig, ax = plt.subplots(1, 1)
# plot_var_cmap(simdps, v.get_x(hpoints['A']))
# ax.plot(-solutions_by_1obj_reindexed[mdps] + solutions_by_1obj_reindexed[mtime])

ndec = 1
xmin = df[o.o_min_mean2degyears_lab].round(ndec).min()
xmax = 150  # df[o.o_min_mean2degyears_lab].round(ndec).max()
x = range(int(xmin * pow(10, ndec)), int(xmax * pow(10, ndec)))
solutions_by_1obj_reindexed = {}
for m in [mdps, mtime]:
    solutions_by_1obj = (
        dfmean[m]
        .groupby(dfmean[m][o.o_min_mean2degyears_lab].round(ndec))[
            o.o_min_cbgemitcost_lab
        ]
        .mean()
    )
    solutions_by_1obj.index = pd.Int64Index(solutions_by_1obj.index * pow(10, ndec))
    solutions_by_1obj_reindexed[m] = solutions_by_1obj.reindex(x).interpolate()


fig = plt.figure()
gs = GridSpec(1, 3, width_ratios=[20, 1, 1])


ax= plt.subplot(gs[0])
df2scat = {}
df2scat[mdps] = dfmean[mdps][dfmean[mdps][o.o_min_mean2degyears_lab]<150]
df2scat[mtime] = dfmean[mtime][dfmean[mtime][o.o_min_mean2degyears_lab]<150]
color_column = o.o_min_cbgedamcost_lab
hue_lims = (0.8,2)
m2cmap = {
    mdps: 'summer',
    mtime: 'winter',
}
norm = mpl.colors.Normalize(vmin=hue_lims[0],
                            vmax=hue_lims[1])

fig = plt.figure()
gs = GridSpec(1, 3)
mlab = mdps
for j, mlab in enumerate([mdps, mtime]):
    cmap = plt.cm.get_cmap(m2cmap[mlab])
    for i, obj in enumerate([o.o_min_mean2degyears_lab, o.o_min_cbgemitcost_lab, o.o_min_cbgedamcost_lab]):
        ax = plt.subplot(gs[i])
        ax.scatter(df2scat[mlab][o.o_min_loss_util_bge_lab],
                   df2scat[mlab][obj],
                   **prop_list[j])

           hue=color_column,
               #size=o.o_min_loss_util_bge_lab,
               data=df2scat[mdps].sort_values(o.o_min_mean2degyears_lab),
               palette=m2cmap[mdps], hue_norm=norm,
               #size=o.o_min_loss_util_bge_lab,
               ax=ax, edgecolor=None)
sb.scatterplot(x=o.o_min_mean2degyears_lab,
               y=o.o_min_cbgemitcost_lab,
               hue=color_column,
               #size=o.o_min_loss_util_bge_lab,
               data=df2scat[mtime].sort_values(o.o_min_mean2degyears_lab),
               palette=m2cmap[mtime], hue_norm=norm,
               #size=o.o_min_loss_util_bge_lab,
               ax=ax, edgecolor=None)

for i, mlab in enumerate([mdps,mtime]):
    cbar_im2a = mpl.colorbar.ColorbarBase(
        ax=plt.subplot(gs[i+1]),
        cmap=cmap,
        norm=norm,
        spacing='uniform',
        orientation='vertical',
        extend='neither')


ax_temp.clear()
for s, sim, pro in zip(["D", "E"], [simtime10k, simdps10k], prop_list):
    sb.distplot(
        temp2100[s].max(),
        ax=ax_temp,
        label=s,
        kde=False,
        norm_hist=True,
        hist_kws={"cumulative": True, "alpha": 0.5},
        **pro
    )

temp_diff = temp2100['D'].loc[:2150].max() - temp2100['E'].loc[:2150].max()
    sb.distplot(
        temp_diff,
        ax=ax_temp_cdf,
        label='Avoided peak',
        kde=True,
        hist=False,
        kde_kws={"lw": 1.5, "cumulative": False},
        **prop_list[0],
    )

temp_over_2k = {}
for s, prop in zip(['D','E'], prop_list):
    temp_over_2k[s] = temp2100[s]-2.5
    temp_over_2k[s] = temp_over_2k[s][temp_over_2k[s]>0].fillna(0).cumsum().loc[2200].mul(5)

sb.boxplot(temp_over_2k['D']-temp_over_2k['E'])
p = 'D'
prop = prop_list[0]
ax_temp.clear()
probscale.probplot(temp_over_2k['D']-temp_over_2k['E'], plottype='prob',
                   probax='x', exceed=False,
                       datalabel='Lognormal Values',  # labels and markers...
                       problabel='',
                       ax=ax_temp,
                       scatter_kws={'markersize': 1},
                       **prop)
ax_temp.set_xlim([4.99,95.01])
ax_temp.set_ylim([-25,15])
ax_temp.invert_yaxis()

tpeak = pd.concat(temp2100).unstack(0).max().unstack(1)

temp_all = pd.concat(temp2100).unstack(0)
temp_all
temp_all[temp_all>2].count()

x = np.linspace(2,4,50)
tcdf = {}
for i, y in enumerate(x):
    tcdf[y] = tpeak[tpeak>y].count()/1e4*100
dftcdf = pd.concat(tcdf).unstack(1)
dftcdf['Difference'] = dftcdf['D']-dftcdf['E']
plt.figure()
(dftcdf['D']/dftcdf['E']).plot()


dam2100: Dict[str, pd.DataFrame] = {}
for s, sim, pro in zip(["E", "D"], [simtime10k, simdps10k], prop_list):
    p = hpoints[s]
    sim.dc.run_and_ret_objs(v.get_x(p))
    dam2100[s] = sim.get("DAMFRAC").loc[2100]

ax_temp_cdf.clear()
import probscale
for p, prop in zip(['E','D'], prop_list):
    probscale.probplot(dam2100[p],
                   probax='x',  # flip the plot
                   bestfit=False,  # draw a best-fit line
                   estimate_ci=False,
                   datalabel='Lognormal Values',  # labels and markers...
                   problabel='Non-exceedance probability',
                   ax=ax_temp_cdf,
                   scatter_kws={'markersize': 1},
                   **prop)
ax_temp_cdf.set_ylim([2,4])
ax.set_ylabel('Normal Values')
ax.set_xlabel('Non-exceedance probability')
ax.set_xlim(left=1, right=99)


ax_temp.legend(title=titleg)
ax_temp.set_xlabel(lab_temp_year(2100))
ax_temp.set_ylabel("PDF")


ax_mitcost.axvline(mitcosts['F'].max(), **prop_list[1])

h = sb.distplot(
    100.0 * (sim.get("ABATECOST").loc[2050].loc[2050])),
    ax=ax_mitcost,
    label=s,
    kde=False,
    norm_hist=True,
    hist_kws={"cumulative": True, "alpha": 0.5},
    **pro,
)

for s, sim, pro in zip(["D", "F"], [simdps10k, simtime10k], prop_list):
    p = hpoints[s]
    sim.dc.run(v.get_x(p))
    if s == "F":
        ax_mitcost.axvline(
            100.0
            * sim.get("ABATECOST").loc[2050].div(sim.get("YGROSS").loc[2050]).mean(),
            **pro,
        )
    else:
        h = sb.distplot(
            100.0 * (sim.get("ABATECOST").loc[2050].div(sim.get("YGROSS").loc[2050])),
            ax=ax_mitcost,
            label=s,
            kde=False,
            norm_hist=True,
            hist_kws={"cumulative": True, "alpha": 0.5},
            **pro,
        )


gs.tight_layout(fig)
