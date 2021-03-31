from collections import namedtuple

from paradoeclim.doeclim import forc2comp
from dicedps.plot.common import *

klist = ['Garner 2016', 'Lamontagne 2018', 'Garner 2018']
klab2code = {
    'Garner 2018': 'k18',
    'Garner 2016': 'ka16',
    'Lamontagne 2018': 'ka18',
}
dcs = {}
for k in klist:
    dcs[k] = h.args2dice(f'-m time -o greg4 -u 1 -w 100 -e 2200 -C {klab2code[k]}')

miu1 = [1.2]*37

from paradoeclim.utils import get_hist_forc_data, get_hist_temp_data
htemp = get_hist_temp_data()
htemp -= htemp.loc[1900:1929].mean()
#htemp.plot()

bau = Data.load('dice_bau', lambda: DiceBase(mode=MODE_OPT).set_bau().run())
bausim=DiceBase(mode=MODE_SIM, vin=['MIU'], setup={'S':bau.S}, endyear=2200).run([0]*37)
#bausim.FORC_year.loc[2015:2020].plot(ax=ax)
hforc = get_hist_forc_data()
alpha2forc = lambda alpha: (hforc[forc2comp['forcing_nonaero']].sum(axis=1)+alpha*hforc[forc2comp['forcing_aero']].sum(axis=1)).values



ecs1 = {}
for adist in u.ClimateSensitivityRV.available_distributions():
    ecs1[adist] = u.ClimateSensitivityRV(adist).cdf(1)
csols=u.ClimateSensitivityRV('olson_unifPrior')
clp()
a = np.linspace(1e-6,1-1e-6,11)
plt.plot(csols.ppf(a), a)
pd.Series(ecs1).plot(kind='bar')

cssamp = u.DoeclimClimateSensitivityUncertainty().levels(50000)
cssamp100 = u.DoeclimClimateSensitivityUncertainty().levels(100)
cslin = np.linspace(cssamp.min(), cssamp.max(), 100)
kappas = [u.get_kappa_alpha_2016(cslin)['kappa'], u.get_kappa_alpha_2018(cslin)['kappa'], u.get_kappa_2018(cslin)['kappa']]
css = [cslin, cslin, cslin]
alphas = [u.get_kappa_alpha_2016(cslin)['alpha'], u.get_kappa_alpha_2018(cslin)['alpha'], 0.5*np.ones_like(cslin)]
labels = ['Garner 2016', 'Lamontagne 2018 (in prep)', 'Garner 2018']




hs = []
fig, axs = plt.subplots(1,2,figsize=(8,3))
for k, l in zip(kappas, labels):
    hs.append(axs[0].plot(cslin, k, label=l, lw=2)[0])
axs[0].set_ylabel('Ocean diffusivity [cm^2/s]')
axs[0].set_xlabel('Climate sensitivity [degC]')
sb.rugplot(cssamp100, ax=axs[0], color='k')
#axs[0].legend()

for a, l in zip(alphas, labels):
    if l == 'Garner 2018':
        ls = '--'
    else:
        ls = '-'
    axs[1].plot(cslin, a, label=l, lw=2, ls=ls)
axs[1].set_ylabel('Aerosol scaling')
axs[1].set_xlabel('Climate sensitivity [degC]')
sb.rugplot(cssamp100, ax=axs[1], color='k')
axs[1].set_ylim([-1,0.7])
axs[1].legend(hs+[plt.Line2D((0,1),(0,0), color='k', ls='-', lw=2)], labels+['Samples from\nNordhaus LogNormal'])
fig.tight_layout()
fig.savefig(incloud('doeclim_calib.png'),dpi=200)




########### Temp & forcing plots - Long-term

clp()

fig, axs = plt.subplots(2,3,figsize=(8,6), sharey='row')
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for ax, k, p in zip(axs.flat, klist, prop_cycle):
    dc = dcs[k]
    dc.run(miu1)
    rel2c = (bget(dc, 'temp')<2).all(0).sum()
    ax.annotate(f'Rel2C = {rel2c}%',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -10), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    y = bget(dc, 'temp')
    y.plot(alpha=0.5, legend=False, ax=ax, **p)
    y2020stats = y.loc[2000].describe()
    ax.annotate(f'{y2020stats["min"]:.2f}-{y2020stats["max"]:.2f} K',
                xy=(2000, y2020stats["min"]),
                xytext=(2, -2), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.set_title(k)
    ax.axhline(2,color='k',ls=':')
    ax.set_xlim([2000, 2200])
    ax.set_ylim([0.5,3])
axs[0,0].set_ylabel('Temperature')


prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for ax, k, p in zip(axs.flat[-3:], klist, prop_cycle):
    dc = dcs[k]
    dc.run(miu1)
    y = bget(dc, 'forcing')
    y.plot(alpha=0.5, legend=False, ax=ax, **p)
    y2020stats = y.loc[2000].describe()
    ax.annotate(f'{y2020stats["min"]:.2f}-{y2020stats["max"]:.2f} Wm-2',
                xy=(2000, y2020stats["min"]),
                xytext=(2, -2), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.set_title(k)
    ax.set_xlim([2000,2200])
    ax.set_ylim([1,4])
axs[1,0].set_ylabel('Forcing')
fig.tight_layout()
fig.savefig(incloud('temp_calib_longterm.png'), dpi=200)


dc = dcs['Garner 2016']
tmp=bget(dc, 'temp')
tmp

x='temp'

######## Temp plot: 1 row
clp()



fig, axs = plt.subplots(1,3,figsize=(8,3), sharey='row')
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for ax, k, p in zip(axs.flat, klist, prop_cycle):
    dc = dcs[k]
    dc.run(miu1)
    rmse = np.sqrt(bget(dc, 'temp').loc[1900:2015].sub(htemp.loc[1900:2015].values, axis=0).pow(2).stack().mean())
    ax.annotate(f'RMSE = {rmse:.2f}',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -10), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    hplo = bplot(dc, 'temp', ax=ax, **p)
    hsca = ax.scatter(htemp.index, htemp.values, color='k', s=5, zorder=100)
    ax.set_title(dc.name)
    ax.axhline(2,color='k',ls=':')
    ax.set_xlim([1900, 2015])
    ax.set_ylim([-0.5,1.5])
#ax.legend([hplo.,hsca],['Model','GISS data'])
axs[0,0].set_ylabel('Temperature')



########### Temp & forcing plots - Hindcast
clp()
fig, axs = plt.subplots(2,3,figsize=(8,6), sharey='row')
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for ax, k, p in zip(axs.flat, klist, prop_cycle):
    dc = dcs[k]
    dc.run(miu1)
    rmse = np.sqrt(bget(dc, 'temp').loc[1900:2015].sub(htemp.loc[1900:2015].values, axis=0).pow(2).stack().mean())
    ax.annotate(f'RMSE = {rmse:.2f}',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -10), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    hplo = bplot(dc, 'temp', ax=ax, **p)
    hsca = ax.scatter(htemp.index, htemp.values, color='k', s=5, zorder=100)
    ax.set_title(k)
    ax.axhline(2,color='k',ls=':')
    ax.set_xlim([1900, 2015])
    ax.set_ylim([-0.5,1.5])
#ax.legend([hplo.,hsca],['Model','GISS data'])
axs[0,0].set_ylabel('Temperature')


prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for ax, k, p in zip(axs.flat[-3:], klist, prop_cycle):
    dc = dcs[k]
    dc.run(miu1)
    hplo = bplot(dc, 'forcing', ax=ax, **p)
    ax.set_title(k)
    ax.set_xlim([1900, 2015])
    #hsca = ax.scatter(hforc.index, alpha2forc(0.3))
    #hsca = ax.errorbar(hforc.index, alpha2forc(0.3), yerr=(alpha2forc(2)-alpha2forc(0)), color='k')
axs[1,0].set_ylabel('Forcing')
fig.tight_layout()
fig.savefig(incloud('temp_calib_hindcast.png'), dpi=200)



##### Forcing DICE & Doeclim

clp()
fig,ax = plt.subplots(1,1,figsize=(6,4))
prop_cycle = iter(plt.rcParams['axes.prop_cycle'])
for p, k in zip(prop_cycle, klist):
    dc = dcs[k]
    dc.run(miu1)
    bget(dc, 'forcing').loc[2010:2020].plot(ax=ax, alpha = 0.5, legend = False, **p)
    dc.forcoth.loc[2015:2020].plot(ax=ax, alpha=0.5, legend=False, **p)
    print(dc.alpha.min(), dc.alpha.max())
alphas = [-0.1, 0.3, 0.5]
for a in alphas:
    y = alpha2forc(a)[-6:]
    ax.annotate(f'alpha = {a:.1f}',xy=(2015,y[-1]), xytext=(0,5), ha='right', va='bottom', textcoords='offset pixels')
    hforcgiss = ax.plot(range(2010,2016), y, 'k', lw=2)
hforcbau_ghg = ax.plot([2015,2020], bau.FORC_year.loc[2015:2020], lw=2, **next(prop_cycle))
ax.annotate(f'DICE climate net forcing', xy=(2015, bau.FORC_year.loc[2015]), xytext=(0, -20), ha='left', va='top', textcoords='offset pixels', arrowprops=dict(arrowstyle="->"))

hforcbau_other = ax.plot([2015,2020], bau.forcoth_year.loc[2015:2020], color='k', lw=2)

dfrcp=pd.DataFrame([[2.089,	2.480],
[2.126,	2.579],
[2.129,	2.584],
[2.154,	2.665]], columns=[2010,2020], index=['rcp6','rcp4.5','rcp2.6','rcp8.5']).T
dfrcp['min'] = dfrcp.min(axis=1)
dfrcp['max'] = dfrcp.max(axis=1)
hrcp = ax.fill_between([2010,2020],dfrcp['min'],dfrcp['max'],alpha=0.5, **next(prop_cycle))
ax.annotate(f'RCPs', xy=(2010, dfrcp.mean(1).loc[2010]), xytext=(10, -20), ha='left', va='top', textcoords='offset pixels', arrowprops=dict(arrowstyle="->"))

ax.errorbar(2011,2.3,yerr=np.array([[2.3-1.1],[3.3-2.3]]), color='k', zorder=100, fmt='o', capsize=5)
ax.annotate(f'IPCC', xy=(2011, 2.3), xytext=(20, -20), ha='left', va='top', textcoords='offset pixels', arrowprops=dict(arrowstyle="->"))

#hforcdc_other = ax.plot([2015,2020], dc.forcoth.loc[2015:2020], **next(prop_cycle))
ax.annotate(f'Exogenous forcing in DICE\nused to offset purple line\n(DICE net forcing)\nto align to previous\nDoeclim forcing time series', xy=(2015, bau.forcoth_year.loc[2015]), xytext=(-10, 0), ha='right', va='bottom', textcoords='offset pixels')
ax.set_ylabel('Forcing')
ax.set_xlabel('Year')
fig.tight_layout()
fig.savefig(incloud('dice_doeclim.png'), dpi=200)
plt.show()
dcb.run_and_ret_objs(miu1)
