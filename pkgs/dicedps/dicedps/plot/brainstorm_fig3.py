from dicedps.plot.common import *

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# https://gist.github.com/ruxi/ff0e9255d74a3c187667627214e1f5fa

def jointplot_w_hue(data, x, y, hue=None, colormap=None,
                    figsize=None, fig=None, scatter_kws=None):
    # defaults
    if colormap is None:
        colormap = sb.color_palette()  # ['blue','orange']
    if figsize is None:
        figsize = (5, 5)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if scatter_kws is None:
        scatter_kws = dict(alpha=0.4, lw=1)

    # derived variables
    if hue is None:
        return "use normal sb.jointplot"
    hue_groups = data[hue].unique()

    subdata = dict()
    colors = dict()

    active_colormap = colormap[0: len(hue_groups)]
    legend_mapping = []
    for hue_grp, color in zip(hue_groups, active_colormap):
        legend_entry = mpatches.Patch(color=color, label=hue_grp)
        legend_mapping.append(legend_entry)

        subdata[hue_grp] = data[data[hue] == hue_grp]
        colors[hue_grp] = color

    # canvas setup
    grid = gridspec.GridSpec(2, 2,
                             width_ratios=[4, 1],
                             height_ratios=[1, 4],
                             hspace=0, wspace=0
                             )
    ax_main = plt.subplot(grid[1, 0])
    ax_xhist = plt.subplot(grid[0, 0], sharex=ax_main)
    ax_yhist = plt.subplot(grid[1, 1])  # , sharey=ax_main)

    ## plotting

    # histplot x-axis
    for hue_grp in hue_groups:
        sb.distplot(subdata[hue_grp][x], color=colors[hue_grp]
                     , ax=ax_xhist)

    # histplot y-axis
    for hue_grp in hue_groups:
        sb.distplot(subdata[hue_grp][y], color=colors[hue_grp]
                     , ax=ax_yhist, vertical=True)

        # main scatterplot
    # note: must be after the histplots else ax_yhist messes up
    for hue_grp in hue_groups:
        sb.regplot(data=subdata[hue_grp], fit_reg=False,
                    x=x, y=y, ax=ax_main, color=colors[hue_grp]
                    , scatter_kws=scatter_kws
                    )

        # despine
    for myax in [ax_yhist, ax_xhist]:
        sb.despine(ax=myax, bottom=False, top=True, left=False, right=True
                    , trim=False)
        plt.setp(myax.get_xticklabels(), visible=False)
        plt.setp(myax.get_yticklabels(), visible=False)

    # topright
    ax_legend = plt.subplot(grid[0, 1])  # , sharey=ax_main)
    plt.setp(ax_legend.get_xticklabels(), visible=False)
    plt.setp(ax_legend.get_yticklabels(), visible=False)

    ax_legend.legend(handles=legend_mapping)

    return dict(fig=fig, gridspec=grid)

import time
def copy2clip(fig):
    from PyQt5.QtGui import QPixmap, QScreen, QImage
    from PyQt5.QtWidgets import QApplication, QWidget
    # c = FigureCanvas(fig)
    if 'qt' == plt.get_backend()[:2].lower():
        fig.canvas.draw()
        pixmap = QWidget.grab(fig.canvas)
        QApplication.clipboard().setPixmap(pixmap)
    # else:
    #    buf = io.BytesIO()
    #    fig.savefig(buf)
    #    QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))


df: pd.DataFrame = load_rerun()

scen2sim = {}
for i, scen in enumerate(zip(['low', 'high'], ['1', '3'])):
    (cli, damfunc) = scen
    scen2sim[scen] = get_sim2plot(mdps, nsow=100, cli=cli, damfunc=damfunc, obj='v3')


ytemp = {}
ymiu = {}
ytempdiff = {}
yabatecost = {}
dfnom = df.xs(mdps, 0, 'miulab').xs('med', 0, 'climcalib').xs('1', 0, 'damfunc')
for i, scen in enumerate(zip(['low', 'high'], ['1', '3'])):
    (cli, damfunc) = scen
    dfcurr = df.xs(mdps, 0, 'miulab').xs(cli, 0, 'climcalib').xs(damfunc, 0, 'damfunc')
    sol = dfnom[np.isclose(dfnom[o.o_min_cbgemitcost_lab], 0.5, atol=1e-3)].sort_values(o.o_min_q95damcost_lab).iloc[0]
    simdps = scen2sim[scen]
    print(simdps.dc.run_and_ret_objs(v.get_x(sol)))
    ytemp[scen] = simdps.get('TATM').round(3).shift(1)
    ymiu[scen] = simdps.get('MIU').round(2).mul(100)
    ytempdiff[scen] = ytemp[scen].diff().round(2)
    yabatecost[scen] = simdps.get('ABATECOST')

prop_list2 = [prop_list[i] for i in [2,4,3,5,6]]+[{'color':'0.5'}]

dictdf2joinplot = {}
for i, scen in enumerate(zip(['low', 'high'], ['1', '3'])):
    (cli, damfunc) = scen
    t = 2030
    dictdf2joinplot[scen] = (
        pd.concat([ytemp[scen].stack(),
                   ytempdiff[scen].stack(),
                   ymiu[scen].stack()],
                  keys=['Temperature (K)', '°C/5yr', 'Abatement (%)'],
                        axis=1).dropna().loc[t].reset_index().sort_values('°C/5yr'))  # 'Abatement (%)', ascending=False)
df2jointplot = pd.concat(dictdf2joinplot, names=['scen','df']).reset_index()
plt.close('all')
fig = jointplot_w_hue(data=df2jointplot,
                          x='Temperature (K)',
                          y='Abatement (%)',
                          colormap=[prop_list[x]['color'] for x in [3,2]],
                          hue='scen')['fig']
copy2clip(fig)


sb.catplot(x='t', y=0, hue='level_0', kind='point', data=pd.concat(yabatecost).stack().reset_index())
