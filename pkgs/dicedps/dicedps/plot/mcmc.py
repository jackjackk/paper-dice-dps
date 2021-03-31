import itertools

from dicedps.plot.common import *

###

niter = int(10e6)
nthin = int(1e3)

inclouddir = lambda *x: os.path.join(os.environ['HOME'], 'CloudStation', 'psu', 'projects', 'brick', *x)
inresdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'brick', 'results', *x)
inplotdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'learn-brick', 'presentations', 'figures', *x)

f2lab = {
    'brick_mcmc_ftony_suninf_e2009_t1870_o4_n10000000': 'BRICK vanilla',
    'doeclim_mcmc_fgreg_suninf_e2009_t1929_o4_n10000000': 'DOECLIM + f.greg',
    'doeclim_mcmc_fgreg_suninf_e2015_t1929_o4_n10000000': 'DOECLIM + f.greg + t.2015',
    'doeclim_mcmc_fgreg_suninf_e2015_t1929_o10_n10000000': 'DOECLIM + f.greg + t.2015 + od.10',
    'doeclim_mcmc_ftony_suninf_e2009_t1929_o10_n10000000': 'DOECLIM + od.10',
    'doeclim_mcmc_ftony_suninf_e2009_t1929_o4_n10000000': 'DOECLIM',
    'brick_mcmc_ftony_sinf_e2009_t1870_o4_n10000000': 'BRICK(inf) vanilla',
    'doeclim_mcmc_ftony_sinf_e2009_t1929_o4_n10000000': 'DOECLIM(inf)',
    'doeclim_mcmc_ftony_sinf_e2009_t1929_o10_n10000000': 'DOECLIM(inf) + od.10',
    'doeclim_mcmc_fgreg_sinf_e2009_t1929_o4_n10000000': 'DOECLIM(inf) + f.greg',
    'doeclim_mcmc_fgreg_sinf_e2015_t1929_o4_n10000000': 'DOECLIM(inf) + f.greg + t.2015',
    'doeclim_mcmc_fgreg_sinf_e2015_t1929_o10_n10000000': 'DOECLIM(inf) + f.greg + t.2015 + od.10',
}

f2lab = {
f'brick_mcmc_fgiss_sinf_t18802009_z19001929_o4_n{niter}': '4) 3 + Switch to NASA GISS forcings',
f'brick_mcmc_fgiss_sinf_t18802011_z19001929_o4_n{niter}': '5) 4 + Extend horizon to 2011',
f'brick_mcmc_fgiss_sinf_t18802015_z19001929_o4_n{niter}': '6) 5 + Extend horizon to 2015',
f'brick_mcmc_furban_sinf_t18502009_z18501870_o4_n{niter}': '1) BRICK default',
f'brick_mcmc_furban_sinf_t18502009_z19001929_o4_n{niter}': '2) 1 + Change normalization period',
f'brick_mcmc_furban_sinf_t18802009_z19001929_o4_n{niter}': '3) 2 + Start in 1880',
f'brick_mcmc_fgiss_sinf_t18802015_z19001929_o10_n{niter}': '7) 6 + Increase OD upper bound',
'brick_mcmc_furban_sinf_t18802009_z19001929_o4_n1000000': '7) 6 w/ less iterations',
'brick_mcmc_fgiss_sinf_t18802011_z19001929_o10_h100_n10000000': '7) H0 +/- 50',
'brick_mcmc_fgiss_sinf_t18802011_z19001929_o4_h150_n10000000': '6) 5 + Extend H0 prior',
'brick_mcmc_fgiss_sinf_t18802011_z19001929_o10_h150_n10000000': '7) 6 + Extend OD prior',
}

def f2lab_default(x):
    return (x.replace('_n10000000','')
             .replace('_h150','')
             .replace('brick_mcmc_','')
             .replace('_t18802009_z18801900','')
             .replace('_t18802011_z18801900','')
             .replace('Ogour',''))

def f2index(f):
    fparts = f.split('_')
    froot = '_'.join(fparts[:-3])
    fburn = int(fparts[-3][1:])
    fthin = int(fparts[-2][1:])
    fnchain = int(fparts[-1][1:])
    return (f2lab.get(froot, f2lab_default(froot)),fburn,fthin,fnchain)


def patt2df(flist, addcalib=False, add41=False):
    _df = {}
    if isinstance(flist, str):
        flist = [flist]
    for fkey in flist:
        for f in tqdm(glob(inresdir(fkey+'_*.nc'))):
            _df[f2index(os.path.basename(f)[:-3])] = u.ncbrick2pandas(f, columns=None)
    if add41:
        dfbnds = pd.read_csv(inresdir('bounds.csv'), index_col=0)
        df41calib = u.ncbrick2pandas(inclouddir('BRICK_postcalibratedParameters_fd-gamma_08May2017.nc'), columns=None)
        df41calib['rho.simple'] = np.nan * df41calib['S']
        _df[('41-param calibration',5,1000,1)] = df41calib[dfbnds.index]
    df = pd.concat(_df).stack().reset_index()
    df.columns = ['run', 'burnin', 'thin', 'nchain', 'nens', 'param', 'value']

    if addcalib:
        cssamp = u.DoeclimClimateSensitivityUncertainty('olson_informPrior').levels(9500)
        kadict = u.get_kappa_alpha_2018(cssamp)
        df2 = pd.DataFrame({lcs: cssamp, lod: kadict['kappa'], las: kadict['alpha']}).stack().reset_index()
        df2.columns = ['nens', 'param', 'value']
        df2['run'] = 'Garner 2018'
        df2['burnin'] = np.nan
        df2['thin'] = np.nan
        df2['nchain'] = np.nan
        df2 = df2[['run', 'burnin', 'thin', 'nchain', 'nens', 'param', 'value']]
        df = pd.concat([df, df2], ignore_index=True)

    return df


#df = patt2df(f2lab.keys())

df = patt2df('*_T*')
dfi = df.set_index(['run', 'burnin', 'thin', 'nchain', 'nens', 'param'])
#inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','meeting-keller-20180504','buffer',*x)

### Boxplot
prior2sample = {}
y = np.zeros(9500)
xprop = np.random.uniform(0.01, 10, 9500)
xpdf = ss.cauchy.pdf(xprop, loc=3, scale=2)
yunif = np.random.uniform(0, max(xpdf), 9500)
prior2sample['sinf'] = xprop[yunif<=xpdf]
prior2sample['slognorm'] = ss.lognorm.rvs(size=1000, s=0.3523, loc=0, scale=np.exp(1.2672))
my_mean = 1.7997
my_std = 0.5242
myclip_a = 0.01
myclip_b = 10.
a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
stn = ss.truncnorm(a, b, my_mean, my_std)
prior2sample['schylek'] = stn.rvs(size=1000)
csprior2lab = {
    'sinf': 'CS0 ~ Cauchy(3,2)',
    'slognorm': 'CS0 ~ Paleosens 2012',
    'schylek': 'CS0 ~ Chylek 2008'
}
odprior2lab = {
    'o4': 'OD0 ~ U[0,4]',
    'o10': 'OD0 ~ U[0,10]'
}
ydefault = dfi.xs('furban_Thadcrut_sinf_o4',0,'run').xs('S',0,'param')['value']
fig, axs = plt.subplots(1,4,figsize=(16,8), sharey=True)
for ax, p in zip(axs, ['sinf_o4', 'sinf_o10', 'slognorm_o4', 'schylek_o4']):
    runlist = [f'{x}_{p}' for x in ['furban_Thadcrut', 'furban_ThadcrutOcheng', 'fgiss_Tgiss', 'fgiss_TgissOcheng']]
    ax.fill_between([0.5,5.5],[1.5,1.5],[4.5,4.5],alpha=0.3)
    data = [prior2sample[p.split('_')[0]],]+[dfi.xs(x,0,'run').xs('S',0,'param')['value'] if x in dfi.index.levels[0] else ydefault*np.nan for x in runlist]
    #data = [prior2sample[p.split('_')[0]], ] + [dfi.xs(x, 0, 'run').xs('S', 0, 'param')['value'] for x in runlist if x in dfi.index.levels[0]]
    ax.boxplot(data, medianprops=dict(linewidth=2))
    ax.set_xticklabels(['Prior',]+['\n'.join(x.split('_')[:2]) for x in runlist], rotation=45, ha='right')
    cs,od = p.split('_')
    ax.set_title(f'{csprior2lab[cs]}, {odprior2lab[od]}')
axs[0].set_ylabel('Climate sensitivity [K]')
axs[0].set_ylim([-0.1,10.1])
fig.tight_layout()
fig.savefig(inplot('boxplot.pdf'))


### AR5 distributions
fig,axs = plt.subplots(8,3, figsize=(8,12), sharey=True, sharex=True)
for d, ax in zip(u.ClimateSensitivityRV.available_distributions(), axs.flat):
    csrv = u.ClimateSensitivityRV(d)
    csrv.plot(ax=ax, color='k', alpha=0.4)
    ax.annotate(d, xy=(0.9, 0.8), xycoords='axes fraction', ha='right')
    #csrv.pdf(xcs)
    #
ax.set_xlim([0,10])
fig.tight_layout()
fig.savefig(inplot('ipcc_distributions.png'),dpi=200)


axs[0].get_xlim()

### Big plot
a = df.run.str.split('_s', expand=True)
a.columns = ['data','priors']
df = pd.concat([df,a], axis=1)

dfcs = df[df.param == 'S']
dfcstab = dfcs.set_index(['data','priors'], append=True).swaplevel(0,-1).sortlevel(axis=0)

list_data = ['fgiss_Tgiss',
             'fgiss_Thadcrut',
             'furban_Tgiss',
             'furban_Thadcrut',
             'furban_ThadcrutOcheng',
             'fgiss_TgissOcheng']
#dfcs['data'].unique()
data2color = dict(zip(list_data, prop_list))


prilab2ls = {'inf_o4':'-', 'lognorm_o4':'--', 'inf_o10':':', 'chylek_o4':'-.'}
list_priors = list(prilab2ls.keys())

grey_means = []
fig,ax = plt.subplots(1,1, figsize=(12,6))
ax.axvspan(1.5, 4.5, alpha=0.1)
hlist = []
llist = []
for d in u.ClimateSensitivityRV.available_distributions():
    csrv = u.ClimateSensitivityRV(d)
    h = csrv.plot(ax=ax, color='grey', alpha=0.4, ls='--')
    #grey_means.append((csrv.data['cs'] * csrv.data['pdf']).sum() / csrv.data['pdf'].sum())
    grey_means.append(pd.Series(csrv.data['pdf'].values, index=csrv.data['cs'].values).argmax())
ax.set_xlim([0,10])
hlist.append(h)
llist.append('IPCC AR5')

# if True: #any('(inf)' in x for x in args):
pprior = lambda ax, x, y, **kws: ax.plot(x, y, lw=2, color='k', **kws) # **prop_list[2])
xcs = np.linspace(0.1, 10, 100).tolist()
my_mean = 1.7997
my_std = 0.5242
myclip_a = 0.01
myclip_b = 10.
a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
"""
for prior, ls in zip([ss.cauchy(loc=3, scale=2), ss.lognorm(scale=np.exp(1.2672), s=0.3523), ss.truncnorm(a, b, my_mean, my_std)],['-','--','-.']):
    h = prior.cdf(10) - prior.cdf(0.1)
    ycs = (prior.pdf(xcs) / h).tolist()
    pprior(ax, xcs, ycs, ls=ls)
"""

means = []
for src, prilab in itertools.product(list_data, list_priors):
    y = dfcstab.xs(src,0,'data').xs(prilab,0,'priors')
    zorder=100
    lw = 2.5
    if (src == 'fgiss_Tgiss') and (prilab == 'chylek_o4'):
        color = prop_list[0]['color']
    elif (src == 'fgiss_Tgiss') and (prilab == 'lognorm_o4'):
        color = prop_list[1]['color']
    elif (src == 'fgiss_Tgiss') and (prilab == 'inf_o10'):
        color = prop_list[2]['color']
    else:
        color = '0.4'
        zorder=None
        lw = 2
    if y.shape[0] != 9500:
        print(f'No {src} {prilab} found')
    #means.append(y['value'].mean())
    sb.distplot(y['value'].values, hist=False, color=data2color[src]['color'],
                kde_kws=dict(label=f'{src} - {prilab}', lw=lw, ls='-', color=color, zorder=zorder)) #ls=prilab2ls[prilab]))
    means.append(pd.Series(ax.lines[-1].get_ydata(), index=ax.lines[-1].get_xdata()).argmax())
    #ax.plot([means[-1],means[-1]],[0,-0.08],color=data2color[src]['color'],lw=3.5,ls=prilab2ls[prilab])
    h = ax.plot([means[-1], means[-1]], [0, -0.06], color=color, lw=lw, ls='-',zorder=zorder)
    if color != '0.4':
        hlist.append(h)
        llist.append(f'{src} - {prilab}')
hlist.append(h)
llist.append('This work')
ax.axhline(0,0,8, color='k')
ax.set_xlim([0,8])
ax.set_ylim([-0.1,1.5])
for g in grey_means:
    #ax.plot([g, g], [0,-0.08], color='grey', lw=1.5)
    ax.plot([g, g], [0, -0.04], color='grey', lw=2)
ax.legend(handles=[h[0] for h in hlist], labels=llist, loc='upper right')
ax.set_xlabel('Climate sensitivity [K]')
ax.set_ylabel('PDF')
#sb.rugplot(means, ax=ax, color='k')
fig.tight_layout()
fig.savefig(inplot('dists.pdf'))
fig.savefig('dists.png',dpi=200)
dfbnds = pd.read_csv(inrootdir('bounds.csv'), index_col=0)

## Plot priors + calibration in BRICK defualt
param2label = {
    'S': 'Climate sensitivity [K]',
    'kappa.doeclim': 'Ocean heat diffusivity [cm^2 s^-1]',
    'alpha.doeclim': 'Aerosol forcing scaling factor'
}
y = df.loc['furban_Thadcrut_sinf_o4'].loc[5].loc[1000].loc[1]
pprior = lambda ax, x, y, **kws: ax.plot(x, y, lw=2, ls='--', color='grey', **kws) # **prop_list[2])
puniprior = lambda ax, lo, up, **kws: pprior(ax, [lo, lo, up, up], [0, ] + [1 / (up - lo)] * 2 + [0, ], **kws)
fig, axs = plt.subplots(1,3, figsize=(16,8))
for param, ax in zip(['S','kappa.doeclim','alpha.doeclim'], axs):
    prior = ss.cauchy(loc=3, scale=2)
    xcs = np.linspace(0.1, 10, 100).tolist()
    h = prior.cdf(10) - prior.cdf(0.1)
    ycs = (prior.pdf(xcs) / h).tolist()
    if param == 'S':
        pprior(ax, [0.1, ] + xcs + [10, ], [0, ] + ycs + [0, ])
    else:
        puniprior(ax, *dfbnds.loc[param].values)
    sb.distplot(y.xs(param, 0, 'param'), ax=ax, hist=False)
    ax.set_xlabel(param2label[param])







def compare_pdfs(name, *args, g4lim=None, **kws):
    cols = kws.get('col_order', [])
    if len(cols) == 3:
        default_size = 4
        default_legend_out = True
    else:
        default_size = 3
        default_legend_out = False
    default_wrap = min(7, len(cols))
    default_kws = dict(col='param', hue='run', hue_order=list(args),
                       size=default_size, aspect=16/9/2, sharex=False, sharey=False,
                       legend_out=default_legend_out)
    default_kws.update(kws)
    if (not 'row' in default_kws) and (not 'col_wrap' in default_kws) and (default_wrap>0):
        default_kws['col_wrap'] = default_wrap
    g = sb.FacetGrid(data=df, **default_kws)
                     #col='param', col_order=['t2co','kappa','alpha'], hue='run', hue_order=list(args),  #['Garner 2018',] + list(args),
    axs = list(g.axes.flat)
    pprior = lambda iax, x, y: axs[iax].plot(x, y, ls='--', color='grey', zorder=0) #**prop_list[2])
    puniprior = lambda iax, lo, up: pprior(iax, [lo, lo, up, up], [0,]+[1/(up-lo)]*2+[0,])
    #if True: #any('(inf)' in x for x in args):
    prior = ss.cauchy(loc=3, scale=2)
    xcs = np.linspace(0.1, 10, 100).tolist()
    h = prior.cdf(10) - prior.cdf(0.1)
    ycs = (prior.pdf(xcs) / h).tolist()
    g.map(sb.distplot, 'value', hist=False, rug=False)
    for iax, ax in enumerate(axs):
        t = ax.title.get_text()
        print(t)
        """
        if 'param = S' in t:
            if 'uninf' in t:
                puniprior(iax, 0.1, 10)
            elif 'inf' in t:
                pprior(iax, [0.1, ] + xcs + [10, ], [0, ] + ycs + [0, ])
            else:
                puniprior(iax, 0.1, 10)
                pprior(iax, [0.1, ] + xcs + [10, ], [0, ] + ycs + [0, ])
        elif 'param = kappa' in t:
            if 'o4' in t:
                puniprior(iax, 0.1, 4)
            elif 'o10' in t:
                puniprior(iax, 0.1, 10)
            elif 'o50' in t:
                puniprior(iax, 0.1, 50)
            else:
                puniprior(iax, 0.1, 4)
                puniprior(iax, 0.1, 10)
        elif 'param = alpha' in t:
            puniprior(iax, 0, 2)
            """
    for i, ax in enumerate(axs):
        if i==0:
            pprior(iax, [0.1, ] + xcs + [10, ], [0, ] + ycs + [0, ])
        else:
            puniprior(i, *dfbnds.iloc[i].values)
    if (g4lim is None) and (default_wrap>0):
        for i in range(int(len(axs)/default_wrap)):
            axs[i*default_wrap+0].set_ylim(0, 0.4); axs[i*default_wrap+0].set_xlim(-1, 11)
            axs[i*default_wrap+1].set_ylim(0, 0.4); axs[i*default_wrap+1].set_xlim(-1, 11)
            axs[i*default_wrap+2].set_ylim(0,3); axs[i*default_wrap+2].set_xlim(-0.5, 2.5)
    else:
        for ax, ax4lim in zip(axs, list(g4lim.axes.flat)):
            ax.set_xlim(ax4lim.get_xlim())
            ax.set_ylim(ax4lim.get_ylim())
    g.add_legend()
    g.fig.savefig(inplot('mcmc_post'+name+'.pdf'), dpi=200)
    return g

compare_pdfs('brick_defualt', 'furban_Thadcrut_sinf_t18802009_z18501870_o4_h50', col='param', col_order=df['param'].unique()[:3])












compare_pdfs('all16', *df['data'].unique()[1:], hue='data', col='param', row='priors',
             col_order=df['param'].unique()[:3],
             row_order = ['uninf_o50', 'inf_o50', 'uninf_o10', 'inf_o10', 'uninf_o4', 'inf_o4'])

#compare_pdfs('all16', *df.run.unique()[1:], col_order=df.param.unique()[:3])

compare_pdfs('01 focus on DOECLIM alone',  'BRICK vanilla', 'DOECLIM')
#compare_pdfs('02 switch to informed prior for CS', 'BRICK vanilla', 'DOECLIM(inf)')
#compare_pdfs('03 replace forcing with NASA GISS forcing',  'BRICK vanilla', 'DOECLIM(inf) + f.greg')
#compare_pdfs('04 extend time from 2009 to 2015',  'BRICK vanilla', 'DOECLIM(inf) + f.greg + t.2015')
#compare_pdfs('05 extend OD upper bound to 10',  'BRICK vanilla', 'DOECLIM(inf) + f.greg + t.2015 + od.10')

#sb.distplot(df[df.param=='t2co']['value'])

clp()

g = compare_pdfs('a6 Extend horizon to 2011','1) BRICK default', '7) 6 + Extend OD prior')
compare_pdfs('a0 Instrumental vs full calibration','1) BRICK default', '41-param calibration', g4lim=g)
compare_pdfs('a1 Change normalization period','1) BRICK default', '2) 1 + Change normalization period', g4lim=g)
compare_pdfs('a2 Start in 1880','1) BRICK default', '3) 2 + Start in 1880', g4lim=g)
compare_pdfs('a3 Switch to NASA GISS forcings','1) BRICK default', '4) 3 + Switch to NASA GISS forcings', g4lim=g)
compare_pdfs('a4 Extend horizon to 2011','1) BRICK default', '5) 4 + Extend horizon to 2011', g4lim=g)
compare_pdfs('a5 Extend horizon to 2011','1) BRICK default', '6) 5 + Extend H0 prior', g4lim=g)


compare_pdfs('Extend horizon to 2011','1) BRICK default', '7) H0 +/- 50', g4lim=g) 
compare_pdfs('Extend horizon to 2011','1) BRICK default', '7) H0 +/- 100', g4lim=g) 
compare_pdfs('Extend horizon to 2011','1) BRICK default', '7) H0 +/- 150', g4lim=g) 

compare_pdfs('Extend horizon to 2015','1) BRICK default', '6) 5 + Extend horizon to 2015', g4lim=g)
compare_pdfs('Extend horizon to 2015','1) BRICK default', '7) 6 w/ less iterations', g4lim=g)
compare_pdfs('Increase OD to 10','1) BRICK default', '7) 6 + Increase OD upper bound', g4lim=g)

df[df.run == '3) 2 + Start in 1880'].groupby('param')['value'].describe()

"""
df = patt2df('doeclim_mcmc_ftony_e2009_t1929_o4_n10000000_b5_t*_n1')
g = sb.FacetGrid(data=df,
                  row='param', hue='thin',
                 size=2, aspect=16/9, sharex=False, sharey=False)
g.map(sb.distplot, 'value', hist=False, rug=False)
g.add_legend()


df = patt2df('doeclim_mcmc_ftony_e2009_t1929_o4_n10000000_b5_t10000_n*')
g = sb.FacetGrid(data=df,
                  row='param', hue='nchain',
                 size=2, aspect=16/9, sharex=False, sharey=False)
g.map(sb.distplot, 'value', hist=False, rug=False)
g.add_legend()


sb.distplot(,hue='level0',col

print(list(_df.keys()))


# Plot auto-correlation

# Plot posteriors

# Chain together

# Plot hindcast
"""
