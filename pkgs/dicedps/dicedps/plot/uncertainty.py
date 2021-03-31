from collections import namedtuple

from paradoeclim.doeclim import forc2comp
from dicedps.plot.common import *

inplot = lambda x: os.path.join(os.environ['HOME'],'working','presentation-egu2018-dice-dps','figures', x)

cssamp = u.DoeclimClimateSensitivityUncertainty().levels(50000)
cssamp100 = u.DoeclimClimateSensitivityUncertainty().levels(100)
cslin = np.linspace(cssamp.min(), cssamp.max(), 100)
kappas = [u.get_kappa_alpha_2018(cslin)['kappa'],]
alphas = [u.get_kappa_alpha_2018(cslin)['alpha'],]
css = [cslin,]


labels = ['Lamontagne 2018 (in prep)',]

hs = []
axs = []
fig = plt.figure(figsize=(8, 4.5))
axs.append(plt.subplot2grid((2,2), (0,1)))
axs.append(plt.subplot2grid((2,2), (1,1)))
axs.append(plt.subplot2grid((2,2), (0,0), rowspan=2))

for k, l in zip(kappas, labels):
    hs.append(axs[0].plot(cslin, k, label=l, lw=2)[0])
axs[0].set_ylabel('Ocean diffusivity [cm^2/s]')
#axs[0].set_xlabel('Climate sensitivity [degC]')
sb.rugplot(cssamp100, ax=axs[0], color='k')

for a, l in zip(alphas, labels):
    axs[1].plot(cslin, a, label=l, lw=2, ls='-')
axs[1].set_ylabel('Aerosol scaling')
axs[1].set_xlabel('Climate sensitivity [degC]')
sb.rugplot(cssamp100, ax=axs[1], color='k')
axs[1].set_ylim([-1,0.7])

sb.distplot(cssamp, ax=axs[2])
sb.rugplot(cssamp100, ax=axs[2], color='k')
axs[2].set_xlabel('Climate sensitivity [degC]')
axs[2].set_ylabel('Probability density')
axs[2].legend(hs+[plt.Line2D((0,1),(0,0), color='k', ls='-', lw=2)], ['Full distribution', '100 samples'])

for ax in axs:
    ax.set_xlim([0.5,7.5])
fig.tight_layout()

fig.savefig(inplot('climate-uncertainty.pdf'))
