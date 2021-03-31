from dicedps.plot.common import *

data = load_data(ldffig2, lhpoints, lhpoints_df, lsol_opt_1obj)

df = data[ldffig2]
dfxyz = data[lhpoints_df]

#%%

clp()
sb.set_context("paper")
figmain = plt.figure(figsize=(w34col * 1.5, hhalf * 1.5))  # , dpi=dpi4supp)
gs = GridSpec(2, 2, figure=figmain)
ax_pareto1 = plt.subplot(gs[0, :])
ax_miu = plt.subplot(gs[1,0])
ax_temp = plt.subplot(gs[1,1])

figsupp = plt.figure(figsize=(w2col * 1.5, hhalf * 1.5))  # , dpi=dpi4supp)
gs = GridSpec(2, 2, figure=figsupp)
ax_pareto2 = plt.subplot(gs[0, :2])
ax_temp_prob = plt.subplot(gs[1, 1])
ax_mitcost = plt.subplot(gs[1, 0])
#ax_mitcost_cdf = plt.subplot(gs[1, 2])

#ax_cmap = plt.subplot(gs[1, 0])


#%%

getxy = lambda s: data[lhpoints][s][
    [o.o_min_mean2degyears_lab, o.o_min_cbgemitcost_lab]
].values

xa, ya = getxy('X')

# Pairs
df150 = df[
    df[o.o_min_mean2degyears_lab] <= data[lsol_opt_1obj].iloc[1][o.o_min_mean2degyears_lab]
    ]
df150.name = "min"

for ax_pareto in [ax_pareto1, ax_pareto2]:
    ax_pareto.clear()
    plot_objective_pairs(
        df150,
        orows=[o.o_min_cbgemitcost_lab],
        ocols=[o.o_min_mean2degyears_lab],
        axs=ax_pareto,
    )

    """
    for letter in ['A',]:
        ax_pareto.scatter(*getxy(letter), s=20, color="w", edgecolor="k")
        ax_pareto.annotate(
            letter,
            getxy(letter),
            xytext=(10, 0),
            textcoords="offset pixels",
            horizontalalignment="left",
            size=12,
            verticalalignment="center",
        )
    """

    for s, va, ha in zip(
            ['Y', 'Z', 'X'],
            ["bottom", "bottom", "top"],
            ["left", "left", "right"]):
        xhigh, yhigh = getxy(
            s
        )  # y.loc[miu, o.o_min_mean2degyears_lab], y.loc[miu, o.o_min_cbgemitcost_lab]
        # if s!='X':
        #    ax_pareto.plot([xa,xhigh],[ya,yhigh], ls='--',color='k')
        """
        ax_pareto.annotate(
            s,
            xy=(xhigh, yhigh),
            xytext=(10 * (ha == "left") - 10 * (ha == "right"), 10*(va=='bottom') - 10*(va=='top')),
            textcoords="offset pixels",
            va="center",
            ha=ha,
            size=12,
        )"""
        ax_pareto.scatter(xhigh, yhigh, s=20, color="w", edgecolor="k")
    """
    xoff = -50
    ax_pareto.annotate(f'Adaptive',
        xy=getxy('X'), xytext=(xoff-10,-20), textcoords='offset pixels',
        ha='right', va='center')
    ax_pareto.annotate(f' | {getxy('X')[0]:.0f}°C-yr',
        xy=getxy('X'), xytext=(xoff-10,-20), textcoords='offset pixels',
        ha='left', va='center')
    ax_pareto.annotate(f' | {getxy('X')[1]:.1f}%CBGE',
        xy=getxy('X'), xytext=(xoff-10,-35), textcoords='offset pixels',
        ha='left', va='center')
    dlab = f'Adaptive | {getxy('X')[0]:.0f}°C-yr, {getxy('X')[1]:.1f}%CBGE'
    elab = f'Non-adaptive | {getxy('Y')[1]:.1f}%CBGE'
    ax_pareto.annotate(elab,
        xy=getxy('Y'), xytext=(10,10), textcoords='offset pixels',
        ha='left', va='center')
    flab = f'Non-adaptive | {getxy('Z')[0]:.0f}°C-yr'
    ax_pareto.annotate(flab,
        xy=getxy('Z'), xytext=(10,10), textcoords='offset pixels',
        ha='left', va='center')
    """

    #ax_pareto.axvline(guide_horiz_x, xmin=0.1, xmax=0.7, color='k', linestyle='--', alpha=0.3, zorder=-100)
    ax_pareto.set_xscale('linear')
    #ax_pareto.set_xticks([50,60,70,80,90,100])
    ax_pareto.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    #ax_pareto.set_xlim([50, 215])
    ax_pareto.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    ax_pareto.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    # ax_pareto.tick_params(axis='both', which='major', labelsize=8)
    # ax_pareto.tick_params(axis='both', which='minor', labelsize=8)
    # ax_pareto.axvline(highlev, ls='--', color='.5')
    guide_vert_y = (getxy('X')+getxy('Y'))[1]/2.
    guide_horiz_x = (getxy('X')+getxy('Z'))[0]/2.

    pareto_xlim = [53,103]
    pareto_ylim = [1,1.93]
    ax_pareto.plot((pareto_xlim[0], getxy('Y')[0]), [guide_vert_y]*2, color='k', linestyle='--', alpha=0.3, zorder=-100)
    ax_pareto.plot([guide_horiz_x]*2, (pareto_ylim[0], getxy('Z')[1]), color='k', linestyle='--', alpha=0.3, zorder=-100)
    ax_pareto.set_xlim(pareto_xlim)
    ax_pareto.set_ylim(pareto_ylim)

    xoff = -50
    if ax_pareto==ax_pareto2:
        xoff = -80
    ax_pareto.annotate(f'{getxy("X")[1]:.1f}%CBGE',
                       xy=getxy("X"), xytext=(-50, 5), textcoords='offset pixels',
                       ha='right', va='bottom')
    ax_pareto.annotate(f'{getxy("X")[0]:.0f}°C-yr',
                       xy=getxy("X"), xytext=(5, -80), textcoords='offset pixels',
                       ha='left', va='bottom')

    ax_pareto.annotate(f'X',
        xy=getxy('X'), xytext=(-5,-5), textcoords='offset pixels',
        ha='right', va='top')
    ax_pareto.annotate('Y',
        xy=getxy('Y'), xytext=(5,5), textcoords='offset pixels',
        ha='left', va='bottom')
    ax_pareto.annotate('Z',
        xy=getxy('Z'), xytext=(5,5), textcoords='offset pixels',
        ha='left', va='bottom')




#%% miu

miu2040 = lambda s: data[lhpoints_df][lhpoints_miu][s].loc[2040]

ax_miu.clear()
hmiuline = ax_miu.axvline(miu2040('Y'), **prop_list[1])
#hline = ax_miu.axvline(miu2040['Z'], **prop_list[1])
hmiudist = sb.distplot(miu2040('X'), ax=ax_miu, label='Adaptive', kde=True, kde_kws={"shade": True}, hist=False, **prop_list[0])  #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)

hleg, lleg = ax_miu.get_legend_handles_labels()
ax_miu.legend(hleg+[hmiuline,], lleg+['Non-adaptive'], loc='upper left')
ax_miu.set_xlabel('Abatement in 2040 (% base CO2)')
ax_miu.set_ylabel('Probability density')
ax_miu.set_xlim([0,100])
#ax_miu.set_ylim([0,2])
#ax_miu.set_major_locator(ticker.AutoLocator())

#%% temp

temp2100 = lambda s: data[lhpoints_df][lhpoints_temp][s].loc[2100]

ax_temp.clear()
sb.distplot(temp2100('X'), ax=ax_temp, label='Adaptive', kde=True, kde_kws={"shade": True}, hist=False, **prop_list[0])  #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)
sb.distplot(temp2100('Z'), ax=ax_temp, label='Non-adaptive', kde=True, kde_kws={"shade": True}, hist=False, **prop_list[1])  #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)

ax_temp.set_xlabel('Temperature in 2100 (K)')
ax_temp.set_ylabel('Probability density')
#ax_temp.set_ylim([0,2])

#%%

fig=figmain
axlist=[ax_pareto1,ax_miu,ax_temp]

hletters = []
for i, ax in enumerate(axlist):
    sb.despine(fig, ax)
    hletters.append(
        ax.text(
            0.95, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, weight="bold"
        )
    )
fig.tight_layout()

#%% main - savefig

savefig4paper(figmain, "main_02")


#%% supp - mitcost

mitcosts_peak = (pd.concat(dfxyz[lhpoints_mitcosts], names=['scen','t'])
                 .max(axis=0, level=0)
                 .loc[['X','Z']]
                 .stack())
ax_mitcost.clear()
#mitcosts_peak[mitcosts_peak.scen=='D'].quantile(.05)
sb.boxplot(y='scen', x=0, data=mitcosts_peak.reset_index(), ax=ax_mitcost,
           whis=[5, 95], width=0.5, showfliers=False,
           medianprops={'lw':2})

x2 = mitcosts_peak.loc['Z'].max()
x1 = mitcosts_peak.loc['X'].quantile(0.5)
x0 = mitcosts_peak.loc['X'].quantile(0.25)
y0 = 0.6
y1 = y0-0.1
#ax_mitcost.axvline(x0, color='k', alpha=0.3, ls='--', zorder=-100)
#ax_mitcost.axvline(x1, color='k', alpha=0.3, ls='--', zorder=-100)
#ax_mitcost.axvline(x2, color='k', alpha=0.3, ls='--', zorder=-100)

ax_mitcost.plot([x1, x2], [y1]*2, color='k') #, markerfacecolor='white')
ax_mitcost.plot([x0, x2], [y0]*2, color='k') #, markerfacecolor='white')
ax_mitcost.annotate(f'{x2-x0:.1f}', xy=(x0,y0), xytext=(2,2),
                    textcoords='offset pixels', ha='left', va='bottom')
ax_mitcost.annotate(f'{x2-x1:.1f}', xy=(x1,y1), xytext=(2,2),
                    textcoords='offset pixels', ha='left', va='bottom')

ax_mitcost.set_yticklabels(['X','Z'])
ax_mitcost.set_ylabel(None)

ax_mitcost.legend([], [], title='Percentiles')
for x in [5,25,50,75,95]:
    ax_mitcost.annotate(f'{x}th', xy=(mitcosts_peak.loc['X'].quantile(x/100), -0.3), xytext=(0, 2),
                        textcoords='offset pixels', ha='center', va='bottom')
    #if x==5:
    #    ax_mitcost.annotate(f'Percentiles:', xy=(mitcosts_peak.loc['D'].quantile(x / 100), -0.4), xytext=(0, 2),
    #                textcoords='offset pixels', ha='left', va='bottom')
ax_mitcost.set_ylim([1.5,-0.8])
ax_mitcost.set_xlabel("Mitigation cost peak\n(% Gross GDP / year)")


#%% supp - temp_prob

tpeak = (pd.concat(dfxyz[lhpoints_temp], names=['scen','t'])
                 .max(axis=0, level=0).stack()).unstack(0)

x = np.linspace(2, 4, 50)
tcdf = {}
for i, y in enumerate(x):
    tcdf[y] = tpeak[tpeak>y].count()/1e4*100
dftcdf = pd.concat(tcdf).unstack(1)
dftcdf['Difference'] = dftcdf['X']-dftcdf['Y']

ax_temp_prob.clear()
(dftcdf['X']/dftcdf['Y']).plot(ax=ax_temp_prob, label='Probability of exceeding given temperature under X\ndivided by the corresponding probability under Y', **prop_list[0])
ax_temp_prob.legend()
#ax_temp_prob.set_xlim(temp_xlim)
ax_temp_prob.set_xlabel("Temperature (°C)")
ax_temp_prob.set_ylabel('Ratio of exceedence probabilities')



#%% supp - layout

fig=figsupp
axlist=[ax_pareto2, ax_mitcost, ax_temp_prob]

hletters = []
for i, ax in enumerate(axlist):
    sb.despine(fig, ax)
    hletters.append(
        ax.text(
            0.95, 1.0, string.ascii_lowercase[i], transform=ax.transAxes, weight="bold"
        )
    )
fig.tight_layout()


#%% supp - savefig

savefig4paper(figsupp, "supp_xyz")



#%% not used

#     temp2100: Dict[str, pd.DataFrame] = {}
#     for s, lab, sim, pro in zip(['N1', 'A'], ['Non-adaptive', 'Adaptive'], [simtime10k, simdps10k], [prop_list[1],prop_list[0]]):
#         p = hpoints[s]
#         sim.dc.run_and_ret_objs(v.get_x(p))
#         temp2100[s] = sim.get('TATM').loc[2100]
#         sb.distplot(temp2100[s], ax=ax_temp, label=lab, kde=True, kde_kws={"shade": True}, hist=False, **pro) # kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)
# 
#     hleg, lleg = ax_temp.get_legend_handles_labels()
#     ax_temp.legend(list(reversed(hleg)), list(reversed(lleg)))
# 
#     ax_temp.set_xlabel(lab_temp_year(2100))
#     ax_temp.set_ylabel('PDF')
# else:
#     ax_mitcost = ax_temp
#     for s, lab, sim, pro in zip(['N2', 'A'], ['Non-adaptive', 'Adaptive'], [simtime10k, simdps10k], [prop_list[1],prop_list[0]]):
#         p = hpoints[s]
#         sim.dc.run(v.get_x(p))
#         if s == 'N2':
#             ax_mitcost.axvline(p[o.o_min_cbgemitcost_lab], **pro)
#         else:
#             h = sb.distplot(Dice.cbge_mitcost_v1(sim.dc._mlist[1]), ax=ax_mitcost, label=s, kde=True, kde_kws={"shade": True}, hist=False, **pro)  #kde=False, norm_hist=True, hist_kws={'alpha':0.5}, **pro)
#     handles, labels = ax_mitcost.get_legend_handles_labels()
#     handles = [handles[0]] + [plt.Line2D([0], [0], lw=1.5, **prop_list[1])]
#     labels =  ['Adaptive', 'Non-adaptive']
#     ax_mitcost.legend(handles, labels)
#     ax_mitcost.set_xlabel(obj2lab2[o.o_min_cbgemitcost_lab].replace('\n',' '))
#     ax_mitcost.set_ylabel('PDF')
# 
# titleg = "From panel (a)"
# 
# """
# ax_diff.scatter(dfdiff.index.levels[0],dfdiff.xs(mtime,0,1)[o.o_min_cbgemitcost_lab].values-dfdiff.xs(mdps,0,1)[o.o_min_cbgemitcost_lab].values, s=5, color='k')
# ax_diff.set_xlabel(obj2lab2[o.o_min_mean2degyears_lab])
# ax_diff.set_ylabel(f'Value of\ninformation')
# ydiff = dfdiff.loc[highlev][o.o_min_cbgemitcost_lab]
# ax_diff.annotate('N - A', xy=(highlev, ydiff[mtime]-ydiff[mdps]), va='bottom', ha='left')
# ax_diff.scatter([highlev], [ydiff[mtime]-ydiff[mdps]], s=10, **prop_list[2])
# ax_diff.set_xlim([xmin,xmax])
# """
# 
# 
# 
# plot_var_cmap(simdps, v.get_x(hpoints['X']), yy=["MIU"], axs=ax_cmap, pad="7%")
# 
# 
# ax_cmap.set_xlabel("Year")
# ax_cmap.set_xlim([2015, 2150])
# hlist = []
# from matplotlib.lines import Line2D
# 
# cmap = simdps.cmap
# hlist = [
#     Line2D([0], [0], color=cmap(1.0), lw=1.5),
#     Line2D([0], [0], color=cmap(0.0), lw=1.5),
# ]
# dlab = 'X'
# elab = 'Y'
# flab = 'Z'
# llist = [f"{dlab}, high climate sensitivity",
#          f"{dlab}, low climate sensitivity",
#          elab, flab]
# for s, ls in zip(['Y', 'Z'], [":", "--"]):
#     simtime.dc.run(v.get_x(hpoints[s]))
#     hlist.append(ax_cmap.plot(100 * simtime.get("MIU"), lw=1.5, ls=ls, color="k")[0])
# ax_cmap.legend(hlist, llist, title=titleg)
