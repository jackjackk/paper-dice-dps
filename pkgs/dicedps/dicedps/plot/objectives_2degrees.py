from dicedps.plot.common import *

#inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','paper-dice-dps','meetings','20180601-keller-group-update','figures',*x)

dfrel2c = load_merged('greg4b')
df2degy = load_merged('greg4d')

simdps = get_sim2plot(mdps)
simtime = get_sim2plot(mtime)

# * Plot just trade-off
ypar = o.o_max_rel2c_lab
xpar = o.o_min_cbgemitcost_lab
# ---
clp()
fig, ax = plt.subplots(1,1,figsize=(12,6))
plot_objective_pairs(dfrel2c, [ypar], [xpar], ax)
fig.tight_layout()
fig.savefig(inplot('fig_tradeoff.png'), dpi=200)


# * Rescale tradeoff
from matplotlib.gridspec import GridSpec
def plot_grid():
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(4, 2)
    ax_pareto = plt.subplot(gs[:-2, 0])
    ax_parallel = plt.subplot(gs[-2:, 0])
    ax_miu = plt.subplot(gs[:2,1])
    ax_tatm = plt.subplot(gs[2:,1])
    return fig, np.array([ax_pareto, ax_parallel, ax_miu, ax_tatm])
# ---
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
plot_var_cmap(simtime, axs=axs[2:])
plot_parallel(ax=axs[1])
fig.tight_layout()
fig.savefig(inplot('fig_tradeoff_grid.png'), dpi=200)


# * Plot min cost sol
out2_saved = []
out2_props = ['0.5', '0.5', prop_list[1], prop_list[0], {'color':prop_list[0]['color'],'ls':'--'}]
def smooth_amiu(amiu):
    fig, ax = plt.subplots(1, 1)
    samiu = pd.Series(amiu, index=simtime.dc.year.values[1:])
    samiu_smooth = samiu.rolling(5, min_periods=1).mean()
    samiu.plot(ax=ax)
    samiu_smooth.plot(ax=ax)
    return samiu_smooth
amiu, out = get_sol_by_mitcost(dfrel2c.loc[mtime], -0.001, relmax=False, retout=True, atol=1e-3)
amiu = smooth_amiu(amiu)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_min_cost.png'), dpi=200)
out2_saved.append(out2)


# * Plot max rel sol
amiu, out = get_sol_by_mitcost(dfrel2c.loc[mtime], retout=True)
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out2_saved[:1], cback=out2_props, **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_max_rel.png'), dpi=200)
out2_saved.append(out2)


# * Plot 0.8 cost sol
amiu, out = get_sol_by_mitcost(dfrel2c.loc[mtime], -0.8, relmax=True, retout=True, atol=1e-3)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out2_saved[:2], cback=out2_props, **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_mid_cost.png'), dpi=200)
out2_saved.append(out2)
amiu_midcost, out_midcost = amiu, out

# * Plot 0.8 high rel DPS sol
amiu, out = get_sol_by_mitcost(dfrel2c.loc[mdps], -0.8, relmax=True, retout=True, atol=1e-3)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simdps, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out2_saved[:3], cback=out2_props, **prop_list[0])
fig.tight_layout()
fig.savefig(inplot('fig_mid_cost_dps_highrel.png'), dpi=200)
out2_saved.append(out2)


# * Plot 0.8 high rel DPS sol - high cli
simdps2 = get_sim2plot(mdps, cli='high')
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simdps2, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out2_saved[:4], cback=out2_props, ls='--', **prop_list[0])
fig.tight_layout()
fig.savefig(inplot('fig_mid_cost_dps_highrel_highcli.png'), dpi=200)
out2_saved.append(out2)

# * Plot 0.8 high rel DPS sol - high cli
simtime2 = get_sim2plot(mtime, cli='high')
amiu, out = amiu_midcost, out_midcost
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(dfrel2c, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime2, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out2_saved[:5], cback=out2_props, ls='--', **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_mid_cost_highcli.png'), dpi=200)



#########
# * Plot control room
ypar = o.o_min_mean2degyears_lab
xpar = o.o_min_cbgemitcost_lab
# ---
clp()
fig,axs=plot_grid()
plot_objective_pairs(df2degy, [ypar], [xpar], axs[0])
plot_var_cmap(simtime, axs=axs[2:])
plot_parallel(ax=axs[1])
fig.tight_layout()
fig.savefig(inplot('fig_2degy_tradeoff_grid.png'), dpi=200)

# * Plot 0.8 cost sol
out3_saved = out2_saved[:2]
amiu, out = get_sol_by_mitcost(df2degy.loc[mtime], -0.8, relmax=True, retout=True, atol=1e-3, sortby=o.o_min_mean2degyears_lab)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(df2degy, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out3_saved[:2], cback=out2_props, **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_2degy_mid_cost.png'), dpi=200)
out3_saved.append(out2)


# * Plot 0.8 Adaptive
amiu, out = get_sol_by_mitcost(df2degy.loc[mdps], -0.8, relmax=True, retout=True, atol=1e-3, sortby=o.o_min_mean2degyears_lab)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(df2degy, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simdps, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out3_saved[:3], cback=out2_props, **prop_list[0])
fig.tight_layout()
fig.savefig(inplot('fig_2degy_mid_cost_dps.png'), dpi=200)
out3_saved.append(out2)

# * Plot 0.8 Adaptive - High cli
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(df2degy, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simdps2, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out3_saved[:4], cback=out2_props, ls='--', **prop_list[0])
fig.tight_layout()
fig.savefig(inplot('fig_2degy_mid_cost_dps_high_cli.png'), dpi=200)
out3_saved.append(out2)


# * Plot 0.8 - High cli
amiu, out = get_sol_by_mitcost(df2degy.loc[mtime], -0.8, relmax=True, retout=True, atol=1e-3, sortby=o.o_min_mean2degyears_lab)
# --
clp()
fig,axs=plot_grid()
plot_objective_pairs(df2degy, [ypar], [xpar], axs[0])
hsca = axs[0].scatter(out[xpar],out[ypar], s=80, facecolor='none', edgecolor='k', lw=5)
out2 = plot_var_cmap(simtime2, amiu, axs=axs[2:])
plot_parallel(out2, ax=axs[1], back=out3_saved[:5], cback=out2_props, ls='--', **prop_list[1])
fig.tight_layout()
fig.savefig(inplot('fig_2degy_mid_cost_high_cli.png'), dpi=200)
out3_saved.append(out2)

# * Damages




#for i in [2,3]: axs[i].clear()
scaler
plot_parallel(out2, axs[1], **prop_list[1])
fig.tight_layout()
plt.tight_layout()

amiu06_time2 = get_sol_by_mitcost(dfrel2c.loc[mtime], 0.8, False)

pd.DataFrame({'max':amiu06_time,'min':amiu06_time2}).plot()

# Find min-max rel solutions
amiu_relmax = get_sol_by_mitcost(dfrel2c.loc[mdps], 0.8, True)
amiu_relmin = get_sol_by_mitcost(dfrel2c.loc[mdps], 0.8, False)








plot_objective_pairs(df2degy, [o.o_min_mean2degyears_lab], [o.o_min_cbgemitcost_lab])



def get_sol(df, miu='rbfXdX41', osort=o.o_min_cbgemitcost_lab, rel=None, ascending=True):
    y = (df.xs(miu, 0, 'miulab')
       .sort_values(o.o_min_cbgemitcost_lab, ascending=ascending)
       .groupby(o.o_max_rel2c_lab, as_index=False)
       .first()).sort_values(o.o_max_rel2c_lab)
    if rel is None:
        ret = y.iloc[-1]
    else:
        ret = y[np.isclose(y[o.o_max_rel2c_lab], rel)].iloc[-1]
    ret.name = f'rel2c: {ret[o.o_max_rel2c_lab]:.1f}%, mitcost: {ret[o.o_min_cbgemitcost_lab]:.2f}%'
    return ret[v.get_xcols(ret)].dropna()


gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])

plot_var_cmap(simdps, amiu_relmin)

sb.distplot(simdps.dc.TATM.loc[2100])
clp()
df = dfrel2c


