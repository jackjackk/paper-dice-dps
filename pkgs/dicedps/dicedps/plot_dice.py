import re
from collections import OrderedDict
import xarray as xr
import matplotlib as mpl
import matplotlib.pylab as plt
import pandas as pd
import glob
import os
import numpy as np
from paradice.dice import Dice
from paradigm import Data, MODE_OPT, namedtuple
import rhodium as rh
import itertools as it
from .viz_dps_policy import load_last_feather
import logging

from tqdm import tqdm


def misc():
    colors = [list(np.array(list(bytes.fromhex(x['color'][1:])))/255) for x in mpl.rcParams['axes.prop_cycle']]*10
    plt.interactive(True)

    ds = load_last_feather()

    ds.head()
    dsr=ds[ds.miu=='dpsr/serial']
    dsr.columns
    #ds=pd.read_feather(os.path.join(os.environ['HOME'], 'CloudStation','psu','projects','dice-dps','last_results.feather'))
    #if os.path.exists(ncfile):
    #    os.unlink(ncfile)
    #ds.to_netcdf(ncfile)
    ds=ds[ds.control!='dpsk1']
    labprettify_dict = {
        'dps5': 'DPS(i)',
        'dpsk': 'DPS(ii)',
        'dpsr': 'DPS(RBF)',
        'temporal': 'No learning'
    }
    labcolor_dict = {
        'dps5': colors[1],
        'dpsk': colors[2],
        'dpsr': colors[3],
        'temporal': colors[0]
    }
    labcolor = lambda l: labcolor_dict[l.split('/')[0]]
    labprettify = lambda l: labprettify_dict[l.split('/')[0]]


    objs = lambda n: [f'obj{i}' for i in range(n)]
    dvs = lambda n: [f'dv{i}' for i in range(n)]


import seaborn as sb

def fig_pareto2d():
    fobjs = [
        [lambda x: -x.obj1, lambda x: x.obj0],
        [lambda x: x.obj0, lambda x: x.obj1]
    ]
    xlabs = ['Neg NPV Abat Cost (% NPV gross GDP)',
             'Utility']
    zord =  {'dps5': 10,
    'dpsk': 5,
    'dpsr': 3,
    'temporal': 1}
    zorder = lambda x: zord[x.split('/')[0]]
    fig, axs = plt.subplots(1,2,figsize=(6,4), sharey=True)
    for j, (ax, obj) in enumerate(zip(axs, ['simple2', 'greg2'])):
        df = ds[ds.obj==obj].loc[:,['miu',]+objs(2)].copy()
        for i, (x,y) in enumerate(df.groupby('miu').groups.items()):
            ax.scatter(*[f(df.loc[y]) for f in fobjs[j]], color=labcolor(x), label=labprettify(x), s=5, zorder=zorder(x))
        ax.set_xlabel(xlabs[j])
        ax.set_title('Objectives '+'AB'[j])
        ax.annotate('*', xy=(0.9, 0.9), xycoords='axes fraction', size=20)
        if j==0:
            ax.set_ylabel('% SOWs with ΔT(2100) <= 2K')
            ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             'last', 'figures', 'fig_pareto2d.pdf'))
#fig_pareto2d()

def fig_controls_dpsk1():
    df = ds[(ds.obj=='greg2') & (ds.miu=='dpsk1/mpi4')].dropna(1)
    n = len([c for c in df.columns if 'dv' in c])
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    for v in ['median']:
        dfcurr = getattr(df.groupby('obj1'), v)().loc[:,dvs(n)]
        dfcurr.columns = [f'x{i+1}' for i in range(n)]
        dfcurr.plot(ax=ax)
    ax.set_xlabel('% SOWs with ΔT(2100) <= 2K')
    ax.set_ylabel('Value of policy parameter')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_dps_controls2.pdf'))
#fig_controls_dpsk1()


def fig_controls_dpsk():
    df = ds[(ds.obj=='simple2') & (ds.miu=='dpsk/mpi4')].dropna(1)
    n = len([c for c in df.columns if 'dv' in c])
    fig,axs = plt.subplots(2,1,figsize=(6,4), sharex=True)
    for v in ['median']:
        dfcurr = getattr(df.groupby('obj0'), v)().loc[:,dvs(n)]
        dfcurr.columns = [f'x{i+1}' for i in range(n)]
        dfcurr.loc[dfcurr['x4'] < 0.05, 'x5'] = np.nan
        dfcurr.loc[dfcurr['x2'] < 0.05, 'x3'] = np.nan
        dfcurr1 = dfcurr.copy()
        dfcurr2 = dfcurr.copy()
        dfcurr1.loc[:, ['x1', 'x2', 'x4', 'x6']] = np.nan
        dfcurr2.loc[:,['x3','x5']] = np.nan
        dfcurr1.plot(ax=axs[0],legend=False)
        dfcurr2.plot(ax=axs[1],legend=False)
        axs[0].legend(ncol=n//2)
    axs[1].set_xlabel('% SOWs with ΔT(2100) <= 2K')
    axs[0].set_ylabel('Value of policy parameter')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_dps_controls_dpsk.pdf'))
#fig_controls_dpsk()

def fig_controls_dpsk1_filtered():
    df = ds[(ds.obj=='greg2') & (ds.miu=='dpsk1/mpi4')].dropna(1)
    n = len([c for c in df.columns if 'dv' in c])
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    for v in ['median']:
        dfcurr = getattr(df.groupby('obj1'), v)().loc[:,dvs(n)].copy()
        dfcurr.columns = [f'x{i+1}' for i in range(n)]
        dfcurr.loc[dfcurr['x4']<0.05,'x5'] = np.nan
        dfcurr.loc[dfcurr['x2']<0.05,'x3'] = np.nan
        dfcurr.plot(ax=ax)
    ax.set_xlabel('% SOWs with ΔT(2100) <= 2K')
    ax.set_ylabel('Value of policy parameter')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_dps_controls3.pdf'))

def fig_controls_dpsk1_filtered2():
    df = ds[(ds.obj=='simple2') & (ds.miu=='dpsk1/mpi4')].dropna(1)
    n = len([c for c in df.columns if 'dv' in c])
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    for v in ['median']:
        dfcurr = getattr(df.groupby('obj0'), v)().loc[:,dvs(n)].copy()
        dfcurr.columns = [f'x{i+1}' for i in range(n)]
        dfcurr.loc[dfcurr['x4']<0.05,'x5'] = np.nan
        dfcurr.loc[dfcurr['x2']<0.05,'x3'] = np.nan
        dfcurr.plot(ax=ax)
    ax.set_xlabel('% SOWs with ΔT(2100) <= 2K')
    ax.set_ylabel('Value of policy parameter')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_dps_controls4.pdf'))
#fig_controls_dpsk1_filtered2()


def main_plot_dice():
    # Load bau
    bau = Data.load('dice_bau', lambda: Dice(mode=MODE_OPT).run())

    ArgsDummy = namedtuple('argsdummy', ['sows','miu','objs'])
    args = ArgsDummy(sows=10, miu='dps5', objs=2)
    # Sample SOWs
    nsow = args.sows
    import random
    from paradice.dice import ECS
    from paradigm import Time, MODE_SIM, MultiModel
    from .dpsrules import MiuKlausController, MiuPolyController

    random.seed(1)
    nsow=100
    clim_sensi_sows = rh.LogNormalUncertainty('Climate Sensitivity', np.exp(ECS.mu), ECS.sigma).levels(nsow)

    # Build DICE
    objlist = ['MAX_REL2C', 'MIN_NPVMITCOST', 'MAX_UTIL', 'MIN_NPVDAMCOST']
    dice_args = dict(time=Time(start=2015, end=2100, tstep=5), mode=MODE_SIM,
                     setup={'S': bau.S, 't2xco2': clim_sensi_sows},
                     default_sow=nsow)
    dctemp = Dice(vin=['MIU'], sow_setup={'Emission control rate GHGs': 0}, **dice_args)
    simdice = Dice(**dice_args)
    simdice.set_inbound('MIU').set_outbound('TATM')
    controller = MiuKlausController(simdice, default_sow=nsow)
    dc = MultiModel(controller, simdice)



    ax1=plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax3=plt.subplot2grid((2, 2), (1, 1))
    ax2=plt.subplot2grid((2, 2), (1, 0))

    dsdps = ds[(ds.miu=='dpsk1/mpi4') & (ds.obj=='greg2')]
    dstemp = ds[(ds.miu=='temporal/serial') & (ds.obj=='greg2')]
    x10pctrel = dsdps.groupby('obj1').mean().loc[8:12,dvs(5)].mean()
    a=dc.run(x10pctrel)
    dc.d.MIU_year.plot(ax=ax1, color='grey', legend=False, label='DPS')
    x10pctrel_temporal = dstemp.groupby('obj1').mean().loc[8:12,dvs(17)].mean()
    dctemp.run(x10pctrel_temporal)
    dctemp.d.MIU_year.plot(ax=ax1, color='blue', legend=False,label='Temporal')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[-2:], ['DPS', 'Temporal'])
    ax1.set_ylabel('Abatement')
    ax1.set_xlabel('Year')

    sb.distplot(dc.d.TATM_year.loc[2100], color='grey', ax=ax2)
    sb.distplot(dctemp.d.TATM_year.loc[2100], color='blue', ax=ax2)
    ax2.axvline(2)
    ax2.set_xlabel('Temperature in 2100 [K]')

    resdps = dc
    restemp = dctemp.d
    util = lambda m: m.tstep * m.scale1 * m.CEMUTOTPER.sum(0) + m.scale2
    sb.distplot(util(resdps), color='grey', ax=ax3)
    sb.distplot(util(restemp), color='blue', ax=ax3)
    ax3.set_xlabel('Utility')
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_10pct_rel.pdf'))




    ax0=plt.subplot2grid((2, 2), (0, 0))
    ax1=plt.subplot2grid((2, 2), (0, 1))
    ax3=plt.subplot2grid((2, 2), (1, 1))
    ax2=plt.subplot2grid((2, 2), (1, 0))

    sb.distplot(Dice.npv_gdp(dc._mlist[1], 'DAMAGES'), color='grey', ax=ax0)
    sb.distplot(Dice.npv_gdp(dctemp, 'DAMAGES'), color='blue', ax=ax0)
    ax0.set_xlabel('NPV Damages (% gross GDP)')
    sb.distplot(Dice.npv_gdp(dc._mlist[1], 'ABATECOST'), color='grey', ax=ax1)
    ax1.axvline(np.mean(Dice.npv_gdp(dctemp, 'ABATECOST')), color='blue')
    ax1.set_xlabel('NPV Abat cost (% gross GDP)')
    #ax1.set_ylim([0,3])

    sb.distplot(dc.d.TATM_year.loc[2100], color='grey', ax=ax2)
    sb.distplot(dctemp.d.TATM_year.loc[2100], color='blue', ax=ax2)
    ax2.axvline(2)
    ax2.set_xlabel('Temperature in 2100 [K]')

    resdps = dc
    restemp = dctemp.d
    util = lambda m: m.tstep * m.scale1 * m.CEMUTOTPER.sum(0) + m.scale2
    sb.distplot(util(resdps), color='grey', ax=ax3)
    sb.distplot(util(restemp), color='blue', ax=ax3)
    ax3.set_xlabel('Utility')
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_10pct_rel2.pdf'))




    ####### Add poly

    simdice2 = Dice(**dice_args)
    simdice2.set_inbound('MIU').set_outbound('TATM')
    controller2 = MiuPolyController(simdice2, default_sow=nsow)
    dc2 = MultiModel(controller2, simdice2)

    dsdpspoly = ds[(ds.miu=='dps5/serial') & (ds.obj=='simple2')]
    x10pctrelpoly = dsdpspoly.groupby('obj0').mean().loc[8:12,dvs(5)].mean()

    dc2.run(x10pctrelpoly)

    dc2.d.MIU_year.plot()

    ax0=plt.subplot2grid((2, 2), (0, 0))
    ax1=plt.subplot2grid((2, 2), (0, 1))
    ax3=plt.subplot2grid((2, 2), (1, 1))
    ax2=plt.subplot2grid((2, 2), (1, 0))

    sb.distplot(Dice.npv_gdp(dc._mlist[1], 'DAMAGES'), color='grey', ax=ax0)
    sb.distplot(Dice.npv_gdp(dctemp, 'DAMAGES'), color='blue', ax=ax0)
    sb.distplot(Dice.npv_gdp(dc2._mlist[1], 'DAMAGES'), color='red', ax=ax0)
    ax0.set_xlabel('NPV Damages (% gross GDP)')

    sb.distplot(Dice.npv_gdp(dc._mlist[1], 'ABATECOST'), color='grey', ax=ax1)
    ax1.axvline(np.mean(Dice.npv_gdp(dctemp, 'ABATECOST')), color='blue')
    sb.distplot(Dice.npv_gdp(dc2._mlist[1], 'ABATECOST'), color='red', ax=ax1)
    ax1.set_xlabel('NPV Abat cost (% gross GDP)')
    #ax1.set_ylim([0,3])

    sb.distplot(dc.d.TATM_year.loc[2100], color='grey', ax=ax2)
    sb.distplot(dctemp.d.TATM_year.loc[2100], color='blue', ax=ax2)
    sb.distplot(dc2.d.TATM_year.loc[2100], color='red', ax=ax2)
    ax2.axvline(2)
    ax2.set_xlabel('Temperature in 2100 [K]')

    resdps = dc
    restemp = dctemp.d
    util = lambda m: m.tstep * m.scale1 * m.CEMUTOTPER.sum(0) + m.scale2
    sb.distplot(util(dc), color='grey', ax=ax3)
    sb.distplot(util(dctemp.d), color='blue', ax=ax3)
    sb.distplot(util(dc2), color='red', ax=ax3)
    ax3.set_xlabel('Utility')
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_10pct_rel3.pdf'))


    ax1=plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax3=plt.subplot2grid((2, 2), (1, 1))
    ax2=plt.subplot2grid((2, 2), (1, 0))

    dc.d.MIU_year.plot(ax=ax1, color='grey', legend=False, label='DPS(ii)')
    dc2.d.MIU_year.plot(ax=ax1, color='red', alpha=0.7, legend=False, label='DPS(i)')
    dctemp.d.MIU_year.plot(ax=ax1, color='blue', legend=False,label='Temporal')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[-2:-1], ['DPS(i)'])
    ax1.set_ylabel('Abatement')
    ax1.set_xlabel('Year')

    sb.distplot(dc.d.TATM_year.loc[2100], color='grey', ax=ax2)
    sb.distplot(dctemp.d.TATM_year.loc[2100], color='blue', ax=ax2)
    sb.distplot(dc2.d.TATM_year.loc[2100], color='red', ax=ax2)
    ax2.axvline(2)
    ax2.set_xlabel('Temperature in 2100 [K]')

    resdps = dc
    restemp = dctemp.d
    util = lambda m: m.tstep * m.scale1 * m.CEMUTOTPER.sum(0) + m.scale2
    sb.distplot(util(dc), color='grey', ax=ax3)
    sb.distplot(util(dctemp.d), color='blue', ax=ax3)
    sb.distplot(util(dc2), color='red', ax=ax3)
    ax3.set_xlabel('Utility')
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_10pct_rel4.pdf'))



    plt.close('all')


    df = ds[ds.obj=='greg4'].copy()
    df.replace({'miu':{'dps5/serial':'DPS(i)',
                       'dpsk1/mpi4':'DPS(ii)',
                       'dpsk/mpi4':'DPS(iib)',
                       'temporal/serial':'Temporal'}}, inplace=True)
    df.columns = ['control', 'borg', 'obj', 'seed', 'idsol', 'NFE', 'dv0', 'dv1', 'dv10',
           'dv11', 'dv12', 'dv13', 'dv14', 'dv15', 'dv16', 'dv2', 'dv3', 'dv4',
           'dv5', 'dv6', 'dv7', 'dv8', 'dv9', 'Utility', '% SOWs ΔT(2100)<2K', 'NPV Mitigation Costs', 'NPV Damage Costs',
           'miu']
    sb.pairplot(vars=['% SOWs ΔT(2100)<2K', 'Utility', 'NPV Mitigation Costs', 'NPV Damage Costs'], diag_kind='kde',
                data=df,
                hue='miu', size=2, aspect=4/3, palette=['red','purple','grey','blue'],
                hue_order=['DPS(i)', 'DPS(iib)','DPS(ii)', 'Temporal'],
                plot_kws=dict(edgecolor=None, alpha=0.5, s=5))

    plt.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_pareto4d.png'), dpi=200)
    """
    ys = []
    ylabs = []
    for df in glob.glob(os.path.join('examples', 'dice', '*serial*greg2*.csv')):
        ys.append(y)
        ylabs.append(os.path.basename(df)[:-4])
    
    # FIGURE DPS CONTROL
    dsdps = ys[0]
    dstemp = ys[1]
    dsdps['obj1_norm']=dsdps['obj1']/100
    dsdps.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'obj0', 'obj1', 'obj1_norm']
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    (dsdps[dsdps.obj1>10].sort_values('obj1')).loc[:,:'x5'].plot(ax=ax)
    ax.set_xlabel('Increasing reliability of 2K')
    ax.set_ylabel('Value of policy parameter')
    ax.get_xaxis().set_ticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_dps_controls.pdf'))
    
    
    
    sb.pairplot(x_vars=['obj0'],y_vars=['obj1'],data=ds[ds.obj=='simple2'],
    sb.pairplot(x_vars=['obj0'],y_vars=['obj1'],data=ds[ds.obj=='simple2'],
                hue='miu',size=6,
                plot_kws=dict(edgecolor=None, alpha=0.5,))
    plt.legend(loc='upper left')
    
    ds[(ds.miu=='dpsk/mpi4') & (ds.obj=='simple2')]['obj0'].head()
    a=ds.sel(obj='greg2').squeeze().to_dataframe().dropna()
    a.index = [a.index.get_level_values(2), a.index.map('{0[0]}/{0[1]}'.format)]
    
    sb.pairplot(vars=objs(4), kind='reg', diag_kind='kde',
                data=ds[ds.obj=='greg4'],
                hue='miu',
                size=4, legend=False)
    
    
    ds.columns
    ds.query("miu=='dpsk/mpi4' and obj=='greg2'").sort_values('obj1').loc[:,'dv0':'dv9'].reset_index(drop=True).plot()
    """



    dctemp.run([0]*17)
    fig, axs = plt.subplots(1,2, figsize=(6,4))
    sb.distplot(dctemp.t2xco2, ax=axs[0], color='grey')
    axs[0].set_xlabel('Climate sensitivity [K]')
    dctemp.d.TATM_year.plot(ax=axs[1], color='grey', legend=False, alpha=0.5)
    axs[1].set_ylabel('Global temperature in BAU [K]')
    axs[1].set_xlabel('Year')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', 'fig_ecs.pdf'))


    """
    
    
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    s=0
    #for s, ax in enumerate(axs.flat):
        #ax.grid(True)
        for y, lab in zip(ys, ylabs): #[ys[0+s], ys[8+s]]):
            ax.scatter(y.obj0, y.obj1, color=labcolor(lab), alpha=0.7, label=labprettify(lab)) # label=y.index.levels[0][-1]
        xmax, xmin = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        ax.legend()
        if 'simple2' in lab:
            ax.set_xlabel('% SOWs with global temperature increase <= 2K')
            ax.set_ylabel('NPV of mitigation costs (% NPV gross GDP)')
            figname = 'fig_simple2.pdf'
            ax.annotate('*', xy=(xmin*0.92,0), size=20)
        elif 'greg2' in lab:
            ax.set_xlabel('Utility')
            ax.set_ylabel('% SOWs with global temperature increase <= 2K')
            figname = 'fig_greg2.pdf'
            ax.annotate('*', xy=(xmin*1.01, ymin*0.92), size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', figname))
    
    
    
    
    ys = []
    ylabs = []
    
    for df in glob.glob(os.path.join('examples', 'dice', '*serial*greg2*.csv')):
        ys.append(y)
        ylabs.append(os.path.basename(df)[:-4])
    
    labprettify_dict = {
        'dps5': 'DPS(i) - Polynomial w/ rate of change information',
        'dpsk': 'DPS(ii) - Power functions w/ future projection information',
        'temporal': 'No endogenous learning'
    }
    labcolor_dict = {
        'dps5': colors[1],
        'dpsk': colors[2],
        'temporal': colors[0]
    }
    labcolor = lambda l: labcolor_dict[l.split('_')[0]]
    labprettify = lambda l: labprettify_dict[l.split('_')[0]]
    
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    s=0
    #for s, ax in enumerate(axs.flat):
        #ax.grid(True)
        for y, lab in zip(ys, ylabs): #[ys[0+s], ys[8+s]]):
            ax.scatter(y.obj0, y.obj1, color=labcolor(lab), alpha=0.7, label=labprettify(lab)) # label=y.index.levels[0][-1]
        xmax, xmin = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        ax.legend()
        if 'simple2' in lab:
            ax.set_xlabel('% SOWs with global temperature increase <= 2K')
            ax.set_ylabel('NPV of mitigation costs (% NPV gross GDP)')
            figname = 'fig_simple2.pdf'
            ax.annotate('*', xy=(xmin*0.92,0), size=20)
        elif 'greg2' in lab:
            ax.set_xlabel('Utility')
            ax.set_ylabel('% SOWs with global temperature increase <= 2K')
            figname = 'fig_greg2.pdf'
            ax.annotate('*', xy=(xmin*1.01, ymin*0.92), size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ['HOME'], 'working', 'paper-dice-dps', 'presentations',
                             '2017-12-Update-PSU', 'figures', figname))
    len(ys)
    y.shape
    y=ys[0]
    ndv = y.shape[1]-2
    dvs = [f'dv{i}' for i in range(ndv)]
    plt.plot(y[dvs].T.values, color='grey', alpha=0.5)
    yrel = y[y.obj1>90]
    y.obj1
    ycheap = y[y.obj0<1]
    plt.plot(yrel[dvs].T.values, color='black', alpha=0.8)
    plt.plot(ycheap[dvs].T.values, color='yellow', alpha=0.8)
    
    
    import seaborn as sb
    #fig = plt.figure(figsize=(6,4))
    
    
    
    pdice = dc.asproblem()
    mdice = dc.asmodel()
    
    """


if __name__ == '__main__':
    main_plot_dice()