from dicedps.plot.common import *
from paradice.dice import Damages

df = load_rerun('greg4h', orient='min')

csdists = ['low','med','high']

sims = {}
for csd in csdists:
    sims[csd] = get_sim2plot(mtime, 100, cli=csd)

fig, axs = plt.subplots(4,2,figsize=(w2col,2*hhalf))
for icol, amiu in enumerate([np.zeros(47), 1.2*np.ones(47)]):
    for irow, csd in enumerate(csdists):
        plot_var_cmap(sim=sims[csd],amiu=amiu,yy=['TATM'],axs=axs[irow,icol])
fig.tight_layout()
savefig4paper(fig, 'supp_csdist2temp')