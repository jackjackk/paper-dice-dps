from dicedps.plot.common import *

def borg2calibdata():
    dfb = pd.read_csv(inrootdir('borg_calib_table.csv'), sep='[\t ]', header=None)
    dfa=dfb.groupby(0)[[1,2]].mean()
    dfc=dfb.groupby(0)[1].describe()
    dfa.to_csv(os.path.join(os.environ['HOME'],'tools','paradoeclim','paradoeclim','data','borg_calib_20180323.csv'), header=None, sep=' ')
    axs[0].fill_between(dfc.index, dfc['min'], dfc['max'])

ppdata =
g = sb.PairGrid(data=ppdata.iloc[:,:2], hue_kws={'cmap':['Greys']}, size=2.5*3/4., aspect=4/3., )


g = sb.PairGrid(data=ppdata, hue_kws={'cmap':['Greys']}, size=2.5*3/4., aspect=4/3., x_vars=['Climate Sensitivity'], y_vars=['Ocean diffusivity', 'Aerosol scaling'])

from seaborn.distributions import _freedman_diaconis_bins

color_rgb = mpl.colors.colorConverter.to_rgb('k')
colors = [sb.utils.set_hls_values(color_rgb, l=l)
          for l in np.linspace(1, 0., 12)]
cmap = sb.palettes.blend_palette(colors, as_cmap=True)
def myhexplot(x, y, **kwargs):
    x_bins = min(_freedman_diaconis_bins(x), 20)
    y_bins = min(_freedman_diaconis_bins(y), 20)
    gridsize = int(np.mean([x_bins, y_bins]))
    if 'ax' in kwargs:
        kwargs['ax'].hexbin(x, y, gridsize=gridsize, mincnt=1, edgecolor='0.5', lw=0.5, cmap=cmap)
    else:
        plt.hexbin(x, y, gridsize=gridsize, mincnt=1, edgecolor='0.5', lw=0.5, cmap=cmap)
g = sb.PairGrid(data=ppdata, hue_kws={'cmap':['Greys']}, size=2.5*3/4., aspect=4/3.)
g = g.map_diag(sb.kdeplot, lw=2, color='k')
g = g.map_offdiag(myhexplot) #, lw=1, shade=True, cbar=False, shade_last=False)


##### Focus
clp()
fig, axs = plt.subplots(1,2, figsize=(8,4))
ax = axs[0]
overlay_calibcurves(1,0)
myhexplot(ppdata['Climate Sensitivity'], ppdata['Ocean diffusivity'],ax=ax)
ax.set_ylim([0,6])
ax.set_xlabel('Climate sensitivity (K)')
ax.set_ylabel('Ocean diffusivity')

#ax.set_xlim([0,10])
ax = axs[1]
overlay_calibcurves(2,0)
myhexplot(ppdata['Climate Sensitivity'], ppdata['Aerosol scaling'],ax=ax)
ax.set_xlabel('Climate sensitivity (K)')
ax.set_ylabel('Aerosol scaling')

#ax.set_ylim([0,5])
#ax.set_xlim([0,10])

hs = []
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for l, p in zip(labels, prop_cycle):
    hs.append(plt.Line2D((0,1),(0,0), lw=2, ls='-', **p))
hs.append(plt.Line2D((0,1),(0,0), lw=2, ls='-', color='k'))
ax.legend(hs, labels+['BRICK'], fontsize=9)
cssamp = u.DoeclimClimateSensitivityUncertainty().levels(50000)
cslin = np.linspace(cssamp.min(), cssamp.max(), 100)
kappas = [u.get_kappa_alpha_2016(cslin)['kappa'], u.get_kappa_alpha_2018(cslin)['kappa'], u.get_kappa_2018(cslin)['kappa'], u.get_kappa_alpha_borg2018(cslin)['kappa']]
css = [cslin, cslin, cslin, cslin]
alphas = [u.get_kappa_alpha_2016(cslin)['alpha'], u.get_kappa_alpha_2018(cslin)['alpha'], 0.5*np.ones_like(cslin), u.get_kappa_alpha_borg2018(cslin)['alpha']]
labels = ['Garner 2016', 'Lamontagne 2018\n(in prep)', 'Garner 2018', 'My attempt']
vlist = [css, kappas, alphas]

def overlay_calibcurves(row, col):
    #ax = g.axes[row, col]
    #xlim, ylim = ax.get_xlim(), ax.get_ylim()
    prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
    for x, y, l, p in zip(vlist[col], vlist[row], labels, prop_cycle):
        if (l == 'Garner 2018') and ((np.isclose(x,alphas[-1]).all() or np.isclose(y,alphas[-1]).all())):
            ls = '--'
        else:
            ls = '-'
        ax.plot(x, y, label=l, lw=2, ls=ls, **p)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)

for coords in [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]:
    overlay_calibcurves(*coords)

hs = []
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for l, p in zip(labels, prop_cycle):
    hs.append(plt.Line2D((0,1),(0,0), lw=2, ls='-', **p))
hs.append(plt.Line2D((0,1),(0,0), lw=2, ls='-', color='k'))
g.axes[0,0].legend(hs, labels+['BRICK'], fontsize=9)
g.fig.savefig(incloud('brick_and_greg_calib.png'), dpi=200)


