from dicedps.plot.common import *

df = load_merged(orient='min')
k = o.o_min_mean2degyears_lab
kws_sort = dict(by = o.oset2labs[last_oset],
                ascending = False)

dftime = df.loc[mtime]
dftime_thinned = dftime.sort_values(**kws_sort).round({k:0}).groupby(k).first().reset_index()

dfdps = df.loc[mdps]
dfdps_thinned = dfdps.sort_values(**kws_sort).round({k:0}).groupby(k).first().reset_index().iloc[1:]
dfdps_objs = dfdps_thinned[o.oset2labs[last_oset]].T
dftest = dftime_thinned[o.oset2labs[last_oset]].T


fig,ax = plt.subplots(1,1)
dftest = dfdps_thinned[o.oset2labs[last_oset]].T

dps_best = []
for obj in o.oset2labs[last_oset]:
    dps_best.append(dfdps_thinned.sort_values(obj).iloc[[0]])
sol_opt_1obj : pd.DataFrame = pd.concat(dps_best).drop_duplicates()


sol_opt_1obj
simdps = get_sim2plot(mdps, 200)

import matplotlib.transforms as mtransforms


dimtemp = np.arange(0.9,3.5,0.1)
dimdifftemp = np.arange(0.,0.35,0.05)


fig, ax = plt.subplots()

plt.hexbin(dimtemp,dimdifftemp,
(df2scat.groupby([df2scat['temp'].round(1),
                                 df2scat['K/5yr'].round(2)])
                ['Abatement (%)']
                .mean()
                .reindex(pd.MultiIndex.from_product([dimtemp, dimdifftemp],
                                                    names=['temp', 'K/5yr']))
                .interpolate(limit_area='inside')
                .unstack().values.T))

sb.heatmap((df2scat.groupby([df2scat['temp'].round(1),
                             df2scat['K/5yr'].round(2)])
            ['Abatement (%)']
            .mean()
            .reindex(pd.MultiIndex.from_product([dimtemp, dimdifftemp],
                                                names=['temp', 'K/5yr']))
            .interpolate(limit_area='inside')
            .unstack().values.T),
           xticklabels=[f'{x:.1f}' for x in dimtemp],
           yticklabels=[f'{x:.2f}' for x in dimdifftemp],
           cmap='plasma_r', vmin=0, vmax=100,
           ax=ax)
ax.invert_yaxis()
if i == 1:
    break
continue

sb.set_context('paper')
fig = plt.figure(figsize=(w2col*1.5, hhalf*1.5))
gs = GridSpec(4, 12)
cmaplist = ['Blues', 'Greens', 'Oranges']

axs_top = []
axs_bottom = []
for j, t in enumerate([2020,2050,2100]):
    ax = plt.subplot(gs[0, j * 4:(j + 1) * 4])
    axs_top.append(ax)
titles_bottom = [
    'Temperature',
    'GDP loss',
    'Mitigation cost',
    'Damage cost'
]
units_bottom = [
    'Temperature\nK / yr',
    'GDP loss\n% GDP / yr',
    'Mitigation cost\n% GDP / yr',
    'Damage cost\n% GDP / yr'
]

for j in range(4):
    ax: plt.Axes = plt.subplot(gs[3, j*3:(j+1)*3])
    #ax.set_title(titles_bottom[j])
    ax.set_ylabel(units_bottom[j])
    axs_bottom.append(ax)


for i, (idsol, sol) in enumerate(sol_opt_1obj.iterrows()):
    #ax = plt.subplot(gs[0, i*4:(i+1)*4])
    simdps.dc.run(v.get_x(sol))
    #plot_var_cmap(simdps, v.get_x(sol), ['MIU'], axs=ax, barplot=(i==2))

    ytemp = simdps.get('TATM').round(3)
    ygross = simdps.get('YGROSS')
    yfinal = simdps.get('Y')
    ydamcost = simdps.get('DAMFRAC')*100.
    yabatcost = (simdps.get('ABATECOST')/ygross)*100.
    yloss = (1-yfinal/ygross)*100.

    kws = prop_list[i]
    ymiu = simdps.get('MIU').round(2).mul(100)
    ytempdiff = ytemp.diff().round(3)
    df2scat = pd.concat([ytemp.stack(), ytempdiff.stack(), ymiu.stack()], keys=['temp', 'K/5yr', 'Abatement (%)'],
                        axis=1).dropna().loc[2020].reset_index() #.sort_values('Abatement (%)', ascending=False)

    #ax.imshow(    axs[j].imshow(np.vstack((yyd,yyt)).T, cmap=cmap, aspect='auto', origin='bottom'))
    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    #hband = ax.fill_between([ytemp.loc[2020].min(), ytemp.loc[2020].max()], 0, 1, facecolor='0.5', alpha=0.5,
    #                             transform=trans)
    for j, t in enumerate([2020,2050,2100]):
        df2scat = pd.concat([ytemp.stack(), ytempdiff.stack(), ymiu.stack()], keys=['temp', 'K/5yr', 'Abatement (%)'],
                        axis=1).dropna().loc[t].reset_index().sort_values('K/5yr') #'Abatement (%)', ascending=False)
        ax = axs_top[j]
        ax.set_ylim([0,110])
        ax.set_xlabel(f'Temperature in {t} (K)')
        out = sb.scatterplot(x='temp', y='Abatement (%)', hue='K/5yr', data=df2scat, edgecolor='k', ax=ax, rasterized=True, palette=cmaplist[i], legend=False)
    ax.set_xlabel('Temperature (K)')
    #out = sb.scatterplot(x='temp', y='K/5yr', hue='Abatement (%)', data=df2scat, edgecolor=None, ax=ax,
    #                     rasterized=True)
    #ax.set_ylabel('Change in Temperature (K/5yr)')
    ax.set_ylabel('Abatement (%)')
    kws = {'color': mpl.cm.get_cmap(cmaplist[i])(1.)[:3]}
    for j, y in enumerate([ytemp, yloss, yabatcost, ydamcost]):
        ax = axs_bottom[j]
        #ax.fill_between(y.index, y.T.quantile(0.05), y.T.quantile(0.95), alpha=0.3, **kws)
        ax.fill_between(y.index, y.T.quantile(0.25), y.T.quantile(0.75), alpha=0.3, **kws)
        ax.plot(y.index, y.T.quantile(0.5), **kws)


ax_mid = plt.subplot(gs[1:3, :])
plot_parallel(dftest,
              front=sol_opt_1obj[o.oset2labs[last_oset]].T,
              cfront='k', ax=ax_mid, alpha=0.3)

gs.tight_layout(fig, h_pad=1.2, w_pad=1.2)
hletters = []
import string
for i, ax in enumerate(axs_top+[ax_mid]+axs_bottom):
    if ax!=ax_mid:
        for side in ["top", "right"]:
            # Toggle the spine objects
            ax.spines[side].set_visible(False)
    hletters.append(ax.text(0.07, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))
hletters
plt.show()
gs.update(wspace=1.2, hspace=1.2)


fig.savefig(inplot('fig_main_01_best1dim.pdf'))

plot_parallel(dftest, ax=ax)

def plot_grid():
    ax_parallel = plt.subplot(gs[:, 0])
    ax_miu = plt.subplot(gs[0,1])
    ax_tatm = plt.subplot(gs[1,1])
    return fig, np.array([ax_parallel, ax_miu, ax_tatm])

sys.exit(1)

simdps10k = get_sim2plot(mdps, 10_000)
simtime = get_sim2plot(mtime, 100)
simdemo = get_sim2plot(mtime, 1)




from matplotlib.gridspec import GridSpec
def plot_grid():
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(2, 2)
    ax_parallel = plt.subplot(gs[:, 0])
    ax_miu = plt.subplot(gs[0,1])
    ax_tatm = plt.subplot(gs[1,1])
    return fig, np.array([ax_parallel, ax_miu, ax_tatm])

def plot_grid23():
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(2, 4)
    axs = []
    axs.append(plt.subplot(gs[:, :2]))
    axs.append(plt.subplot(gs[0,2]))
    axs.append(plt.subplot(gs[0,3]))
    axs.append(plt.subplot(gs[1,2]))
    axs.append(plt.subplot(gs[1,3]))
    return fig, np.array(axs)


def plot_grid14():
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(2, 4)
    axs = []
    axs.append(plt.subplot(gs[1, :]))
    axs.append(plt.subplot(gs[0,0]))
    axs.append(plt.subplot(gs[0,1]))
    axs.append(plt.subplot(gs[0,2]))
    axs.append(plt.subplot(gs[0,3]))
    return fig, np.array(axs)

dps_best = []
for obj in o.oset2labs[last_oset]:
    dps_best.append(dfdps_thinned.sort_values(obj).iloc[0].name)
time_best = []
for obj in o.oset2labs[last_oset]:
    time_best.append(dftime_thinned.sort_values(obj).iloc[0].name)


dps_best

apoint = dfdps_thinned.iloc[dps_best].iloc[0]
amiu2run = v.get_x(apoint)

simdps10k.dc.run_and_ret_objs(amiu2run)
ytemp = get_variable_from_doeclim(simdps10k.dc, 'temp')
ytemp.loc[2015].describe()
ytemp.T.mean().plot()

simdps.dc.run(amiu2run)
ytemp = simdps.get('TATM').round(3)
ymiu = simdps.get('MIU').round(2).mul(100)
ytemp.head()

ytemp.loc[2015].describe()
ytempdiff = ytemp.diff().round(3)
df2scat = pd.concat([ytemp.stack(), ytempdiff.stack(), ymiu.stack()], keys=['temp','K/5yr','Abatement (%)'],axis=1).dropna().reset_index()

fig, ax = plt.subplots(1, 1)
import matplotlib.transforms as mtransforms
#trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
out = sb.scatterplot(x='temp', y='K/5yr', hue='Abatement (%)', data=df2scat, ax=ax, edgecolor=None, rasterized=True)
hband = ax.fill_between([ytemp.loc[2020].min(), ytemp.loc[2020].max()], *ax.get_ylim(), facecolor='0.5', alpha=0.5, zorder=-10)


clp()
sb.set_context('notebook', font_scale=1.1)

#fig, axs = plot_grid23()
lboth = ['adap','nonadap']


def plot_overview_1dim(what=lboth):
    fig, axs = plot_grid14()
    if what == lboth:
        plot_parallel(
            avec=-dfdps_thinned.loc[dps_best][o.oset2labs[last_oset]].T,
            back=-dftime_thinned.iloc[time_best][o.oset2labs[last_oset]].T,
            cback='0.5', cargs={'ls':'--'}, ax=axs[0],
            arrwidth=0.5, **prop_list[1])
    elif what == ['nonadap']:
        plot_parallel(-dftime_thinned.iloc[time_best][o.oset2labs[last_oset]].T,
                      color='0.5', ls='--', ax=axs[0], arrwidth=0.5)
    hlist = [plt.Line2D([0], [0], lw=1.5, **prop_list[1]),
             plt.Line2D([0], [0], lw=1.5, ls='--', color='0.5')]
    axs[0].legend(hlist, ['Adaptive', 'Non-adaptive'])

    prop_labs = [{'ylabel': True, 'barlabel': False},
    {'ylabel': False, 'barlabel': False},
    {'ylabel': False, 'barlabel': False},
    {'ylabel': False, 'barlabel': True}]
    titles = ['Min 2-deg years', 'Min utility loss', 'Min mit. cost', 'Min dam. cost']

    for iax, ax in enumerate(axs[1:]):
        xmiu = v.get_x(dfdps_thinned.iloc[dps_best].iloc[iax])
        if what == lboth:
            plot_var_cmap(simdps, xmiu, yy=['MIU'], axs=ax, years=[2050,2100,2150], **prop_labs[iax])
        elif what == ['nonadap']:
            plot_var_cmap(simdps, None, yy=['MIU'], axs=ax, years=[2050, 2100, 2150], **prop_labs[iax])
        xmiu = v.get_x(dftime_thinned.iloc[time_best].iloc[iax])
        simtime.dc.run(xmiu)
        y = simtime.get('MIU').T.mean().mul(100.)
        if iax==2:
            y = smooth_amiu(y)
        y.plot(ax=ax, legend=False, lw=2, ls='--', color='0.5')
        ax.set_title(titles[iax])
        ax.set_xlabel('Year')

    fig.tight_layout()
    fig.savefig(inplot(f'fig-parallel-plot-{"-".join(what)}.pdf'))

plot_overview_1dim(['nonadap'])
plot_overview_1dim(lboth)


# ---

#avec = df.loc[mtime,o.oset2labs['greg4d']].head().T

clp()
sb.set_context('notebook', font_scale=1.1)
fig,axs=plot_grid()
dftest = dftime_thinned[o.oset2labs[last_oset]].T
plot_parallel(-dftest, ax=axs[0])
plot_miu_cmap(simdemo, dftime_thinned, axs=axs[1], cbarticks=['300', '200', '100', '0'])
fig.tight_layout()
axs[2].set_visible(False)
fig.savefig(inplot('fig-parallel-plot-1dim-00-space.pdf'))
#axs[0].set_xlim([-0.3,3.])
fig,axs=plot_grid()

sol_util = dftime_thinned.sort_values(o.o_max_util_bge_lab).iloc[0]
sol_util_objs = sol_util[o.oset2labs[last_oset]]
plot_parallel(-dftest, front=-sol_util_objs, cfront='k', ax=axs[0], alpha=0.3)
plot_miu_cmap(simdemo, dftime_thinned, ahigh=sol_util, chigh='k', axs=axs[1], alpha=0.3, cbarticks=['300', '200', '100', '0'])
plot_var_cmap(simtime, v.get_x(sol_util), yy=['TATM'], axs=axs[2])
fig.tight_layout()
fig.savefig(inplot('fig-parallel-plot-1dim-10-utility.pdf'))


fig,axs=plot_grid()
sol_util = dftime_thinned.sort_values(o.o_min_mean2degyears_lab).iloc[0]
sol_util_objs = sol_util[o.oset2labs[last_oset]]
plot_parallel(-dftest, front=-sol_util_objs, cfront='k', ax=axs[0], alpha=0.3)
plot_miu_cmap(simdemo, dftime_thinned, ahigh=sol_util, chigh='k', axs=axs[1], alpha=0.3, cbarticks=['300', '200', '100', '0'])
plot_var_cmap(simtime, v.get_x(sol_util), yy=['TATM'], axs=axs[2])
fig.tight_layout()
fig.savefig(inplot('fig-parallel-plot-1dim-20-2degyears.pdf'))
fig.savefig(inplot('fig-parallel-plot-1dim-30-damcost.pdf'))


fig,axs=plot_grid()
sol_util = dftime_thinned.sort_values(o.o_min_cbgemitcost_lab).iloc[0]
sol_util_objs = sol_util[o.oset2labs[last_oset]]
plot_parallel(-dftest, front=-sol_util_objs, cfront='k', ax=axs[0], alpha=0.3)
plot_miu_cmap(simdemo, dftime_thinned, ahigh=sol_util, chigh='k', axs=axs[1], alpha=0.3, cbarticks=['300', '200', '100', '0'])
plot_var_cmap(simtime, v.get_x(sol_util), yy=['TATM'], axs=axs[2])
fig.tight_layout()
fig.savefig(inplot('fig-parallel-plot-1dim-40-mitcost.pdf'))

objlist = o.oset2labs[last_oset]
for objcurr, p in zip(objlist, prop_list):
    for miucurr, ls in zip([mtime,mdps], ['-','--']):
        break
    break

out_saved = []
out_props = ['0.5', '0.5', prop_list[1], prop_list[0], {'color':prop_list[0]['color'],'ls':'--'}]

miucurr = mtime
# max utility
objcurr = o.o_max_util_bge_lab
amiu = smooth_amiu(v.get_x(df.loc[miucurr].sort_values(objcurr).iloc[0]))
fig,axs=plot_grid()
out = plot_var_cmap(simtime, amiu, axs=axs[1:])
plot_parallel(-out, ax=axs[0], iorder=None, **prop_list[1])
fig.tight_layout()
out_saved.append(-out)
fig.savefig(inplot('fig-parallel-plot-1dim-10-utility.pdf'))

objcurr = o.o_min_mean2degyears_lab
amiu = smooth_amiu(v.get_x(df.loc[miucurr].sort_values(objcurr).iloc[0]))
fig,axs=plot_grid()
out = plot_var_cmap(simtime, amiu, axs=axs[1:])
plot_parallel(-out, ax=axs[0], back=out_saved, cback=out_props, **prop_list[1])
fig.tight_layout()
out_saved.append(-out)
fig.savefig(inplot('fig-parallel-plot-1dim-20-2degyears.pdf'))
fig.savefig(inplot('fig-parallel-plot-1dim-30-damcost.pdf'))

objcurr = o.o_min_cbgemitcost_lab
amiu = smooth_amiu(v.get_x(df.loc[miucurr].sort_values(objcurr).iloc[:10].mean()))
fig,axs=plot_grid()
out = plot_var_cmap(simtime, amiu, axs=axs[1:])
plot_parallel(-out, ax=axs[0], back=out_saved, cback=out_props, **prop_list[1])
fig.tight_layout()
out_saved.append(out)
fig.savefig(inplot('fig-parallel-plot-1dim-40-mitcost.pdf'))


