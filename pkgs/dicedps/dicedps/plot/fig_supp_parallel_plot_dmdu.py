from dicedps.plot.common import *


simdps10k = get_sim2plot(mdps, 10_000)
simdps = get_sim2plot(mdps, 100)
simtime = get_sim2plot(mtime, 100)
simdemo = get_sim2plot(mtime, 1)

df = load_merged(orient='min')
k = o.o_min_mean2degyears_lab
dftime = df.loc[mtime]
dftime_thinned = dftime.sort_values(o.oset2labs[last_oset]).round({k:0}).groupby(k).first().reset_index()

dfdps = df.loc[mdps]
dfdps_thinned = dfdps.sort_values(o.oset2labs[last_oset]).round({k:0}).groupby(k).first().reset_index().iloc[1:]
dfdps_objs = dfdps_thinned[o.oset2labs[last_oset]].T
dftest = dftime_thinned[o.oset2labs[last_oset]].T



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
        plot_parallel(-dfdps_thinned.iloc[dps_best][o.oset2labs[last_oset]].T,
                      back=-dftime_thinned.iloc[time_best][o.oset2labs[last_oset]].T,
                      cback='0.5', cargs={'ls':'--'}, ax=axs[0], arrwidth=0.5, **prop_list[1])
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


