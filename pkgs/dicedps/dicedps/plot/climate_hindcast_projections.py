from dicedps import dice_helper as h
from dicedps.environ import *

import os

nsow = 10000
ncfiles = {}
#ncfiles['test'] = indatabrick("brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n20000000_t1000_b5-chains_t1_b0.nc")
ncfiles['med'] = 'med'
"""ncfiles['Low CS High Conf'] = os.path.join(os.environ['HOME'], 'working', 'brick', 'results', 'brick_mcmc_fgiss_TgissOgour_schylek_t18802011_z18801900_o4_h150_n10000000_b5_t1000_n1.nc')
ncfiles['Med CS Med Conf'] = os.path.join(os.environ['HOME'], 'working', 'brick', 'results', 'brick_mcmc_fgiss_Tgiss_slognorm_t18802011_z18801900_o4_h150_n10000000_b5_t1000_n1.nc')
ncfiles['High CS Low Conf'] = os.path.join(os.environ['HOME'], 'working', 'brick', 'results', 'brick_mcmc_fgiss_Tgiss_sinf_t18802011_z18801900_o10_h150_n10000000_b5_t1000_n1.nc')
"""
dcmap = {}

for cslab, ncf in ncfiles.items():
    dcmap[cslab] = h.args2dice(f'-m time -o greg4 -u 1 -w {nsow} -e 2200 -C {ncf} -t')



p26 = ["S","kappa.doeclim","alpha.doeclim","T0","H0","beta0","V0.gsic","n","Gs0","a.te","b.te","invtau.te","TE0","a.simple","b.simple","alpha.simple","beta.simple","V0","sigma.T","sigma.H","rho.T","rho.H","sigma.gsic","rho.gsic","sigma.simple","rho.simple"]

from paradoeclim import get_hist_temp_data
from dicedps.plot.common import *
miu0 = np.zeros(37)
miu1 = 1.2*np.ones_like(miu0)

# Load data
dfforc = (pd.read_csv(indoeclim('data','forcings.csv'),
                      index_col=[0, 1]).xs('giss',0,'source'))

dcurr = dcmap[list(ncfiles.keys())[0]] #'Low CS High Conf']
dcurr.run(miu1)
a = dcurr.temp
dcurr_faero = pd.Series(dcurr._mlist[3].forcing_aero[1:],index=dcurr._mlist[3].year[1:])
def plot_check_forcings_aero():
    fig,ax=plt.subplots(1,1)
    dfforc['aerosols'].plot(ax=ax)
    dcurr_faero.plot(ax=ax)
(dcurr_faero.loc[:2015]-dfforc.loc[:2015,'aerosols']).sort_values(ascending=False).head()

# Run model
tempmap = {}
for lab, dcurr in dcmap.items():
    dcurr.run(miu1)
    y = get_variable_from_doeclim(dcurr, 'temp').dropna()
    #y = y.sub(y.loc[1880:1920].mean())
    tempmap[lab] = y

#y.sub(y.loc[1880:1920].mean()).loc[1880:1920].mean().describe()

fig0, axs0 = plt.subplots(1,3,figsize=(12,6), sharey='row')
fig1, axs1 = plt.subplots(1,3,figsize=(12,6), sharey='row')
htemp_orig = pd.DataFrame({x: get_hist_temp_data(name=x) for x in ['giss','hadcrut5']})
#htemp = htemp_orig
##Normalization
htemp = htemp_orig.sub(htemp_orig.loc[1880:1920].mean(0))
#for lab, color in zip(htemp.columns, ['r','k']):
#    htemp[lab].reset_index().plot(x='Year',y=lab,kind='scatter',s=5,color=color,ax=ax)
#    break
for i, (lab, p) in enumerate(zip(dcmap.keys(), prop_cycle())):
    ax = axs0[i]
    ax.plot(htemp.index, htemp['giss'], color='k')
    #ax.errorbar(x=htemp.index, y=htemp['hadcrut5'],
    #                           yerr=[np.zeros(len(htemp)), htemp['giss'] - htemp['hadcrut5']], color='k',
    #                           fmt='o', capsize=0, markersize=3)
    y = tempmap[lab]
    ax.set_title(lab)
    ax.set_xlabel('Year')
    rmse = {x: np.sqrt(y.mean(1).loc[1900:2015].sub(htemp.loc[1900:2015,x].values, axis=0).pow(2).mean()) for x in ['giss','hadcrut5']}
    ax.annotate(f'RMSE =\nGISS:     {rmse["giss"]:.2f}\nHADCRUT5: {rmse["hadcrut5"]:.2f}',
                #xy=(2015, y.loc[2015]), xytext=(-100, 0), textcoords='offset pixels',
                xy=(0, 1), xycoords='axes fraction', xytext=(10, -10), textcoords='offset pixels',
                horizontalalignment='left', verticalalignment='top')
    #y.median(1).plot(ax=ax, label=lab, **p)
    ax.fill_between(y.loc[:2020].index, *[y.loc[:2020].apply(x, axis=1) for x in [partial(np.percentile, q=q) for q in [5,95]]], alpha=0.5, **p)
    y.loc[:2020].mean(1).plot(ax=ax, label=lab, ls='-', lw=2, **p)
    nsows4est = 100
    nsows4cent = nsows4est//100
    nestimates = nsow//nsows4est
    rel2c = np.zeros(nestimates)
    for off in range(nestimates):
        rel2c[off] = (y.iloc[:,off::nestimates] < 2).all(0).sum()/nsows4cent
    y.loc[2000:].iloc[:,::nestimates].plot(ax=axs1[i], label=lab, ls='-', lw=3, alpha=0.5, legend=False, **p)
    #rel2c = (y<2).all(0).sum()
    axs1[i].annotate(f'Rel2C = {np.min(rel2c):.0f}-{np.max(rel2c):.0f}%',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(10, -10), textcoords='offset pixels',
                horizontalalignment='left',
                verticalalignment='top')
    ax.set_ylabel('Temperature anomaly [K]')
    axs1[i].set_ylabel('Temperature anomaly [K]')
    axs1[i].set_xlabel('Year')
    axs1[i].set_title(lab)
    axs1[i].axhline(2, ls='--', lw=1, color='0.2', alpha=0.5)
fig0.tight_layout()
fig1.tight_layout()
fig0.savefig(inplot('hindcast.pdf'))
fig1.savefig(inplot('proj2c.pdf'))
