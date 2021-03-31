from dicedps.plot.common import *

#%% load data
vlist = [
    ldfthinned_2degyear_dps,
    lsol_opt_1obj,
    ldfsol1d_reeval
]

data = {}
for vcurr in vlist:
    data[vcurr] = pd.read_parquet(inoutput('dicedps', f'{vcurr}.dat'))


idsol2letter = dict(zip(data[lsol_opt_1obj][o.o_min_mean2degyears_lab].index, 'ABC'))
nsol_opt_1obj, scaler = get_scaled_df(data[lsol_opt_1obj][ocolset], data[ldfthinned_2degyear_dps][ocolset])

labels_abc = [f'{x} ({y})' for x, y in zip('ABC',
                                                ['Min. warming above 2°C', 'Max. utility', 'Min. mitigation cost'])]
colors_abc = plt.get_cmap('viridis_r')(nsol_opt_1obj[o.o_min_mean2degyears_lab][::-1])


#%% init figure grid

clp()
sb.set_context('paper')
fig = plt.figure(figsize=(w2col*1.5, hhalf*1.5))

ncols = 12

outer_grid = gridspec.GridSpec(2,1, height_ratios=[1,2]) # gridspec with two adjacent horizontal cells
top_row = outer_grid[0,0] # the left SubplotSpec within outer_grid
top_row_grid = gridspec.GridSpecFromSubplotSpec(1,3, top_row, wspace=0.3)

axs_top = []
axs_bottom = []
for j, t in enumerate([2020,2050,2100]):
    ax = plt.subplot(top_row_grid[0, j])
    axs_top.append(ax)

ax_mid = plt.subplot(outer_grid[1, 0])

cmaplist = ['Blues', 'Greens', 'Oranges']

units_bottom = [
    'Temperature\n°C / yr',
    'GDP loss\n% GDP / yr',
    'Mitigation cost\n% GDP / yr',
    'Damage cost\n% GDP / yr'
]

fig_supp, axs_supp = plt.subplots(2, 2, figsize=(w2col*1.5, hhalf*1.5))
axs_supp = axs_supp.flatten()
for j in range(4):
    ax: plt.Axes = axs_supp[j]  #plt.subplot(bottom_row_grid[j])
    #ax.set_title(titles_bottom[j])
    ax.set_ylabel(units_bottom[j])
    axs_bottom.append(ax)



#%% parallel plot

ax_mid.clear()

ocolset_pplot = [
    o.o_max_inv_loss_util_bge_lab,
    o.o_min_mean2degyears_lab,
    o.o_min_cbgedamcost_lab,
    o.o_min_cbgemitcost_lab,
]

if data[ldfthinned_2degyear_dps].shape[0] > 2000:
    ax_mid.set_rasterization_zorder(1)

plot_parallel_new(
    ParallelPlotLayer(data=data[ldfthinned_2degyear_dps][ocolset_pplot],
                      kwargs=dict(alpha=0.5, icolor=1)),
    ParallelPlotLayer(data=data[lsol_opt_1obj][ocolset_pplot],
                      kwargs=dict(color='k', labels='ABC')),
    keep_norm_labs=[o.o_max_inv_loss_util_bge_lab],
    invert_min_labs=True,
    ax=ax_mid)

ax_mid.set_ylabel('Objective performance\n(arrow gives preferred direction)')


#%% scatter

plot_scatter_temp_miu(df=data[ldfsol1d_reeval],
                      axs=axs_top,
                      colors_abc=colors_abc,
                      labels_abc=labels_abc)


#%% supp

df = data[ldfsol1d_reeval]

for i, idsol in enumerate(df.index.levels[0]):
    kws = {'color': mpl.cm.get_cmap(cmaplist[i])(1.)[:3]}
    for j, ylab in enumerate([lytemp, lyloss, lyabatcost, lydamcost]):
        ax = axs_bottom[j]
        y = data[ldfsol1d_reeval].loc[idsol][ylab].unstack()
        ax.fill_between(y.index, y.T.quantile(0.25), y.T.quantile(0.75), alpha=0.3, **kws)
        ax.plot(y.index, y.T.quantile(0.5), **kws)
        letter2write = idsol2letter[idsol]
        if j == 2:
            _overrule_letter = {
                'A': 'A/B',
                'B': ''
            }
            letter2write = _overrule_letter.get(letter2write, letter2write)
        ax.annotate(letter2write,
                    xy=(y.index[-1], y.iloc[-1].mean()),
                    xytext=(5, 0), textcoords='offset pixels',
                    va='center', ha='left')
        # letter2tpeak[idsol2letter[idsol]] = ytemp.T.quantile(0.5).max()
    # pd.Series(letter2tpeak)

handles_abc2 = [plt.Line2D(range(1), range(1), markersize=8, color=x(1)[:3], ls='-', lw=3) for x in
                [mpl.cm.Blues_r, mpl.cm.Greens_r, mpl.cm.Reds_r]]


axs_bottom[0].legend(
    handles_abc2 + [plt.Line2D(range(1), range(1), color='k', ls='-', lw=1.5),
                    plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='k', alpha=0.3)],
    labels_abc + ['Median', '25th-75th\npercentile']
)

#%% final layout and lettering

outer_grid.tight_layout(fig, h_pad=1.5, pad=1) #, pad=0, h_pad=0, w_pad=-1) #, h_pad=1.0, w_pad=1.0) #, rect=(0, 0, 1, 1))

hletters = []
import string
for i, ax in enumerate(axs_top+[ax_mid]): #+axs_bottom):
    if ax!=ax_mid:
        for side in ["top", "right"]:
            # Toggle the spine objects
            ax.spines[side].set_visible(False)
        letter_x = 0.07
    else:
        letter_x = 0.0
    hletters.append(ax.text(letter_x, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))


fig_supp.tight_layout(h_pad=1.5, pad=1)

hletters = []
import string
for i, ax in enumerate(axs_bottom):
    if ax!=ax_mid:
        for side in ["top", "right"]:
            # Toggle the spine objects
            ax.spines[side].set_visible(False)
        letter_x = 0.07
    else:
        letter_x = 0.0
    hletters.append(ax.text(letter_x, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, weight='bold'))


#%% save file

fig.savefig(inplot('fig_main_01_best1dim.pdf'), dpi=200)

fig_supp.savefig(inplot('fig_supp_01_best1dim.pdf'))

