from collections import defaultdict

from dicedps.plot.common import *


cscalib = _cli2calib

inplot = lambda *x: join(home, 'working', 'paper-dice-dps', 'meetings', 'figures', *x)

miulist = ['rbfXdX44','time']

# Load pareto
os.chdir(os.path.join(os.environ['HOME'], 'working/dicedps/sandbox'))
d = v.load_pareto('*greg4b*_c*merged.csv', revert_max_objs=False, objlabs=False)
#df = pd.concat({miu:dforig.loc[miu].loc[con] for miu, con in zip(miulist,['none','inertmax'])}, axis=0)

sols = defaultdict(dict)
for miu in miulist:
    y = df.loc[miu].reset_index(level='nfe',drop=True).sort_values(o.o_max_rel2c)
    #sols['mindam'][miu] = y.loc[y[o.o_min_cbgedamcost].idxmin()]
    #sols['minmit'][miu] = y.loc[y[o.o_min_cbgemitcost].idxmin()]
    #sols['maxrel'][miu] = y.loc[y[o.o_max_rel2c].idxmin()]
    #sols['maxutil'][miu] = y.loc[y[o.o_max_util_bge].idxmin()]
    for xmit in [0.,0.25,0.5,0.75,1]:
        yy = y[np.isclose(y[o.o_min_cbgemitcost], xmit, atol=1e-2)].iloc[-1]
        sols[f'mit{xmit:.1f}'][miu] = yy
#solmindam = sols['mindam']
#pd.DataFrame(solmindam)


dictreeval = {}
listcalibfiles = [cscalib[x] for x in ['low','med','high']] #[os.path.basename(x)[:-3] for x in glob(inrootdir('dicedps','data','brick*nc'))]
for miu in miulist:
    currlist = defaultdict(list)
    for ncfile in tqdm(listcalibfiles):
        dc = h.args2dice(f'{q.miu2arg(miu)} -c doeclim -u 1 -w 1000 -e 2250 -s 1 -o greg4b -C {ncfile} -t -S 1')
        for solname in tqdm(list(sols.keys())):
            y = sols[solname][miu]
            x = y[v.get_xcols(y)].dropna()
            currlist[solname].append(pd.Series(dc.run_and_ret_objs(x)))
    dictreeval[miu] = pd.concat({solname: pd.concat(currlist[solname], axis=0, keys=listcalibfiles) for solname in sols.keys()})
dictreeval[miu]
dfreeval = pd.concat(dictreeval).unstack(-1)
dfreeval.to_csv('dfreeval.csv')
dfreeval = o.normalize2min(pd.read_csv('dfreeval.csv', index_col=[0,1,2]))
ocols = v.get_ocols(dfreeval)
#ocols_labs = [o.obj2lab[x] for x in ocols]

ndfreeval = get_scaled_df(dfreeval[ocols], df[ocols])

simdps = get_sim2plot(miulist[0])
simtime = get_sim2plot(miulist[1])
for solname, sol in sols.items():
    y = sol[miulist[0]]
    x = y[v.get_xcols(y)].dropna()
    fig, axs = plot_var_cmap(simdps,['MIU'],x)
    miu = miulist[1]
    y = sol[miulist[1]]
    x = y[v.get_xcols(y)].dropna()
    simtime.dc.run(x)
    simtime.get('MIU').mean(1).plot(ax=axs[0],color='0.1',ls='-.',lw=2)
    fig.savefig(inplot(f'miu_{solname}.pdf'))


for i, (solname, sol) in enumerate(sols.items()):
    fig, axs = plt.subplots(1,3,sharey=True, figsize=(6,2))
    for j, (ax, cs) in enumerate(zip(axs, listcalibfiles)):
        for miu, p in zip(miulist, prop_list):
            ndfreeval.loc[miu].loc[solname].loc[cs].T.plot(ax=ax, legend=False, label=miu2lab[miu], **p)
            #ndfreeval.loc[miu].loc[cscalib['med']].T.plot(ax=ax, legend=False, ls='--', **p)
        ax.set_xticks(range(4))
        if (i==0) and (j==2):
            ax.legend()
        #ax.set_xticklabels(ocols)
    #fig.tight_layout()
    fig.savefig(inplot(f'parallel_plot_{i}_{solname}.pdf'))

# Run
a = dc.run_parallel(dfx.iterrows(), ncpus=args.ncpus)

# Find interesting solutions, e.g. min DAM

fig, axs = plt.subplots(1, 3, figsize=(6,2))
for ifocus, ax in enumerate(axs):
    for i, ((k, ncfile), p) in enumerate(zip(cscalib.items(), prop_list)):
        sdata = u.ncbrick2pandas(inrootdir('dicedps','data',ncfile+'.nc'), columns=None)['S']
        if i == ifocus:
            color='0.0'
        else:
            color='0.7'
        sb.distplot(sdata, ax=ax, color=color)
    ax.set_xlabel('Climate sensitivity [K]')
fig.tight_layout()
fig.savefig(inplot(f'cs_dists.pdf'))
