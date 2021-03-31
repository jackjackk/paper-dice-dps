from dicedps.plot.common import *


#df = v.load_pareto_mpi('u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#v.save_pareto(df, 'u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#df[v.get_ocols(df)]

df = v.load_pareto('*med*greg4d*_c*rerun.csv', revert_max_objs=False)
inplot = lambda *x: join(home, 'working', 'paper-dice-dps', 'inertia', 'figures', *x)
simdps = get_sim2plot('rbfXdX41')
simtime = get_sim2plot('time')


# Why does the high rel2C constrained-by-design outperforms the inter-temoporal?
def get_cheap_rel_sol(miu='rbfXdX41', osort=o.o_min_cbgemitcost_lab, rel=None, ascending=True):
    y = (df.xs(miu, 0, 'miulab')
       .sort_values(o.o_min_cbgemitcost_lab, ascending=ascending)
       .groupby(o.o_max_rel2c_lab, as_index=False)
       .first()).sort_values(o.o_max_rel2c_lab)
    if rel is None:
        ret = y.iloc[-1]
    else:
        ret = y[np.isclose(y[o.o_max_rel2c_lab], rel)].iloc[-1]
    ret.name = f'rel2c: {ret[o.o_max_rel2c_lab]:.1f}%, mitcost: {ret[o.o_min_cbgemitcost_lab]:.2f}%'
    return ret[v.get_xcols(ret)].dropna()

miu='time'
amiu = get_cheap_rel_sol('time', ascending=False)
plot_var_cmap(simtime, ['MIU','TATM'], amiu)

amiu = get_cheap_rel_sol(ascending=False)
plot_var_cmap(simdps, ['MIU','TATM'], amiu)

fig, ax = plt.subplots(1,1)
dfdps = df.xs('rbfXdX41', 0, 'miulab')
ax.scatter(dfdps['dv20'],dfdps[o.o_max_rel2c_lab])

# How does cheap high rel inter-temporal sol look like?
amiu = get_cheap_rel_sol('time', 'inertmax', ascending=True)
dct.run(amiu)
fig, axs = plt.subplots(2,1,sharex=True, figsize=(7,4))
for ax, y, lab in zip(axs,
                 [dct.MIU, (dct.MIU-dct.MIU.shift()).div(5)],
                 ['MIU', 'Change in MIU']):
    y.loc[:2200].plot(ax=ax,legend=False)
    ax.set_ylabel(lab)
#fig.tight_layout()
fig.savefig(inplot('time_maxrel_miu.png'), dpi=150)


# What about correpsonding temp?

#fig.tight_layout()
simtime = get_sim2plot('time')

amiu = get_cheap_rel_sol(simtime.miu, 'inertmax')
fig, ax = plot_var_cmap(simtime, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'time_maxrel.pdf'))


for asc, lab in zip([True, False], ['cheap','exp']):
    amiu = get_cheap_rel_sol(simdps.miu, 'inertmax', ascending=asc)
    fig, ax = plot_var_cmap(simdps, ['MIU', 'TATM'], amiu)
    fig.savefig(inplot(f'dps_maxrel_{lab}.pdf'))

amiu = get_cheap_rel_sol(simdps.miu, 'inert95q', ascending=False)
fig, ax = plot_var_cmap(simdps, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_maxrel_95q.pdf'))


amiu = get_cheap_rel_sol(simtime.miu, 'inertmax', rel=22)
fig, ax = plot_var_cmap(simtime, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'time_rel20.pdf'))

amiu = get_cheap_rel_sol(simdps.miu, 'none', rel=22)
fig, ax = plot_var_cmap(simdps, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_noconstr_rel20.pdf'))

amiu = get_cheap_rel_sol(simdps.miu, 'none', rel=22, ascending=False)
fig, ax = plot_var_cmap(simdps, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_noconstr_rel20_exp.pdf'))

amiu = get_cheap_rel_sol(simtime.miu, 'none', rel=22.1)
fig, ax = plot_var_cmap(simtime, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'time_noconstr_rel20.pdf'))

simdps4 = get_sim2plot('rbfXdX44')

amiu = get_cheap_rel_sol(simdps4.miu, 'none', rel=22.4, ascending=True)
fig, ax = plot_var_cmap(simdps4, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_hybrid_rel20.pdf'))

amiu = get_cheap_rel_sol(simdps4.miu, 'none', rel=22.4, ascending=False)
fig, ax = plot_var_cmap(simdps4, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_hybrid_rel20_exp.pdf'))

amiu = get_cheap_rel_sol(simdps4.miu, 'none')
fig, ax = plot_var_cmap(simdps4, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_hybrid_maxrel.pdf'))

amiu = get_cheap_rel_sol(simdps4.miu, 'none', ascending=False)
fig, ax = plot_var_cmap(simdps4, ['MIU', 'TATM'], amiu)
fig.savefig(inplot(f'dps_hybrid_maxrel_exp.pdf'))

fig, ax = plt.subplots(1,1,figsize=(4,4))
for i, (sim, con, p) in enumerate(zip([simdps4, simdps, simtime], ['none','inertmax','inertmax'], prop_list)):
    y = (df.loc[sim.miu].loc[con]
       .sort_values(o.o_min_cbgemitcost_lab, ascending=True)
       .groupby(o.o_max_rel2c_lab, as_index=True))[o.o_min_cbgemitcost_lab]
    ymin = y.first()
    ymax = y.last()
    ax.plot(ymin.index, ymin.values, alpha=0.5, **p)
    ax.plot(ymax.index, ymax.values, alpha=0.5, **p)
    ax.fill_between(ymin.index, ymin.values, ymax.values, label=miu2lab[sim.miu], zorder=i, alpha=0.5, **p)
ax.legend()
ax.set_xlabel('Reliability 2C (% SOWs)')
ax.set_ylabel('Mitigation cost (% today\'s consumption)')
fig.tight_layout()
fig.savefig(inplot('tradeoff_rel2c_mitcost.pdf'))


fig, axs = plt.subplots(1,2,figsize=(6,4))
for ax, xcol, xlab in zip(axs,
                          [o.o_max_rel2c_lab, o.o_min_cbgedamcost_lab],
                          ['Reliability 2C (% SOWs)',
                           'Damage cost (% today\'s consumption)']
                          ):
    for i, (sim, con, p) in enumerate(zip([simdps, simtime], ['none','inertmax'], [prop_list[1],prop_list[0]])):
        y = df.loc[sim.miu].loc[con]
        ax.scatter(y[xcol], y[o.o_min_cbgemitcost_lab], alpha=0.2,
                   edgecolor=mpl.colors.colorConverter.to_rgba(p['color'], alpha=1),
                   label=miu2lab[sim.miu], **p)
        """
        y = (df.loc[sim.miu].loc[con]
           .sort_values(o.o_min_cbgemitcost_lab, ascending=True)
           .groupby(o.o_min_cbgedamcost_lab, as_index=True))[o.o_min_cbgemitcost_lab]
        ymin = y.first()
        ymax = y.last()
        ax.scatter(ymin.index, ymin.values, alpha=0.5, **p)
        ax.scatter(ymax.index, ymax.values, alpha=0.5, **p)
        #ax.fill_between(ymin.index, ymin.values, ymax.values, label=miu2lab[sim.miu], zorder=i, alpha=0.5, **p)
        """
    ax.legend()
    ax.set_xlabel(xlab)
    ax.set_ylabel('Mitigation cost (% today\'s consumption)')
fig.tight_layout()
fig.savefig(inplot('tradeoff_mitcost2.pdf'))







amiu = get_cheap_rel_sol('rbfXdX04', 'inertmax', osort=o.o_max_util_bge_lab) #, rel=20.2)
dcx.run(amiu)
fig, ax = plot_var_cmap(dcx, 'MIU')
ax.set_ylabel('MIU')




def plot2ax(ax, con, color=True, osort=o.o_min_cbgemitcost_lab):
    if color:
        plist = prop_list
    else:
        plist = [{'color': '0.5', 'alpha':0.3}]*2
    for x, p in zip(df.index.levels[0], plist):
        try:
            (df.loc[x].loc[con]
             .sort_values(osort)
             .groupby(o.o_max_rel2c_lab)
             .first()[osort]
             .plot(ax=ax, label=x, lw=2, **p))
        except:
            pass


# Rel2c vs cost
fig,axs=plt.subplots(1,3, sharex=True, sharey=True)
conlist = df.index.get_level_values('con').unique()
for ax, con in zip(axs, conlist):
    for con2 in conlist:
        if con2==con:
            continue
        #plot2ax(ax, con2, color=False, osort=o.o_min_npvmitcost_lab)
        plot2ax(ax, con2, color=False)
    plot2ax(ax, con, color=True)
    ax.legend()
    ax.set_title(con)



#
dc = {
    'x4': dice_greg4b('-m XdX4'),
    'x': dice_greg4b('-m XdX'),
    't':
}


css = dc['x4'].t2co.sort_values()
isorted = css.index
colors = mpl.cm.cool(scaler.fit_transform(css.values.reshape(-1,1), css).flatten())

#a = get_cheap_rel_sol('rbfXdX44')
dc['x4'].run(a)
dc['x4'].TATM[isorted].rename(columns={i:f'{x:.2f}' for i, x in css.items()}).plot(color=colors)

dc.run(a)
for ax, x in zip(plt.subplots(1,2)[1], [dc.MIU,dc.TATM]):
    x.plot(ax=ax)

dcx4 = h.args2dice('-e 2250 -m XdX4 -r 4 -S 1 -w ')
dcx = h.args2dice('-e 2250 -m XdX -r 4 -S 1 ')

df_sorted_rel2c = df.xs('inertmax',0,'con')

y = df.loc['time'].loc['none']
(y.sort_values(o.o_max_util_bge_lab)
 .iloc[-1][v.get_xcols(y)]).plot()


y = df.loc['time'].loc['inertmax']
(y.sort_values(o.o_min_npvmitcost_lab)
 .groupby(y[o.o_max_rel2c_lab].round(0))
 .first()[v.get_xcols(y)]
 .rename(columns={f'dv{x}':x for x in range(48)})
 .T.plot(colormap='Greens'))
# .stack().reset_index()).plot.scatter(c=o.o_max_rel2c_lab,x='level_1',y=0)

y_max_rel2c = df_sorted_rel2c.iloc[0]
y_max_rel2c.plot()


dc = dice_greg4('-x inertmax -m XdX')
df_sorted_rel2c = df.loc[mdps].xs('inertmax',0,'con').sort_values(o.o_max_rel2c_lab, ascending=False)
y_max_rel2c = df_sorted_rel2c.iloc[0]
dc.run(y_max_rel2c[v.get_xcols(y_max_rel2c)].dropna())
dc.MIU.plot()
sb.distplot(dc.t2co)

df = v.load_pareto(flist, objlabs=False)
df.head()
ocols = o.oset2vout['jack5']
ocols_round = []
for x in [o.o_max_rel2c,o.o_min_peakabatrate]:
    df[x+'_round'] = df[x].round(1)
    ocols_round.append(x+'_round')
df.sort_values(,ascending=[x[:3]=='MIN' for x in ocols]).head()

# Inertia vs rel2c
ax.set_xlim([0,19.4])

# Rel2c vs cost
fig,ax=plt.subplots(1,1)
for x in [mtime,mdps]:
    df.loc[x].sort_values(o.o_min_npvmitcost).groupby([ocols_round[0]]).first()[o.o_min_npvmitcost].plot(ax=ax, label=x)
ax.legend()

# Dam vs cost
fig,ax=plt.subplots(1,1)
for x in [mtime,mdps]:
    y = df.loc[x].sort_values(o.o_min_npvmitcost)
    y.groupby(y[o.o_min_npvdamcost].round(2)).first()[o.o_min_npvmitcost].plot(ax=ax, label=x)
ax.legend()

os.chdir(os.path.join(os.environ['HOME'], 'working', 'dicedps', 'sandbox'))
df = v.load_pareto('*_merged.csv')
dc = {}
colsmiu = {}
for x in [mtime,mdps]:
    dc[x] = h.args2dice(f'{q.miu2arg(x)} -u 1 -w 100 -e 2200 -o jack5 -C brick_fgiss_tgiss_scauchy_o4 -t')
    colsmiu[x] = [f'dv{i}' for i in range(len(dc[x].get_bounds()))]
inertia=4
sols_5inertia = df[np.isclose(df[o.o_min_peakabatrate_lab],inertia,rtol=1e-2)].sort_values(o.o_max_rel2c_lab).groupby('miu').last()

######## Plot inertia issue
fig, axs = plt.subplots(1,2,figsize=(8,4))
ax=axs[0]
for x, p in zip([mdps, mtime], prop_list):
    #for x, p in zip([mtime, mdps], [prop_list[1],prop_list[0]]):
    y = df.loc[x].sort_values([o.o_max_rel2c_lab,o.o_min_npvmitcost_lab])
    yy = y.groupby(y[o.o_min_peakabatrate_lab].round(2)).last()
    ax.scatter(yy[o.o_min_peakabatrate_lab], yy[o.o_max_rel2c_lab], label=miu2lab[x], alpha=0.5, **p)
ax.legend()
ax.set_ylabel('Reliability 2C (%)')
ax.set_xlabel('Max peak abatement rate (%/yr)')
ax.axvline(inertia, color='0.',lw=1.5,ls='--')
ax=axs[1]
ax.set_xlim([2010.75, 2104.25])
xx = np.arange(2015,2105,5)
ax.plot(xx,np.minimum((xx-2015)/5*0.2,1), ls='--', color='0.', lw=1.5)
ax.set_ylim([0,1.3])
ax.set_xlabel('Year')
ax.set_ylabel('Abatement (1=100%)')
fig.tight_layout()
fig.savefig(inplot('inertia0.pdf'))

x=mtime
yy = sols_5inertia.loc[x]
p=prop_list[1]
axs[0].scatter(yy[o.o_min_peakabatrate_lab], yy[o.o_max_rel2c_lab], edgecolor='k', lw=1, **p)
run = dc[x].run(sols_5inertia.loc[x][colsmiu[x]])
run.MIU.loc[:2100].plot(ax=ax, legend=False, **p)
ax.set_xlabel('Year')
fig.savefig(inplot('inertia1.pdf'))

x=mdps
yy = sols_5inertia.loc[x]
p=prop_list[0]
axs[0].scatter(yy[o.o_min_peakabatrate_lab], yy[o.o_max_rel2c_lab], edgecolor='k', lw=1, **p)
run = dc[x].run(sols_5inertia.loc[x][colsmiu[x]])
run.MIU.loc[:2100].plot(ax=ax, legend=False, **p)
ax.plot(xx+10,np.maximum(np.minimum((xx-2015)/5*0.2,1),0), ls='--', color='0.', lw=1.5)
ax.plot(xx+10,np.maximum(np.minimum((xx-2015)/5*0.1,1),0), ls='--', color='0.', lw=1.5)
ax.plot(xx+10,np.maximum(np.minimum((xx-2015)/5*0.15,1),0), ls='--', color='0.', lw=1.5)
ax.set_xlabel('Year')
fig.savefig(inplot('inertia2.pdf'))


##### Plot inertia after 150
fig, ax = plt.subplots(1,1,figsize=(8,4))

x=mtime
yy = sols_5inertia.loc[x]
p=prop_list[1]
run = dc[x].run(sols_5inertia.loc[x][colsmiu[x]])
run.MIU.loc[:2175].plot(ax=ax, legend=False, lw=2, **p)

x=mdps
yy = sols_5inertia.loc[x]
p=prop_list[0]
run = dc[x].run(sols_5inertia.loc[x][colsmiu[x]])
run.MIU.loc[:2175].plot(ax=ax, legend=False, lw=2, **p)
ax.set_xlabel('Year')
ax.set_ylabel('Abatement (1=100%)')
fig.tight_layout()
fig.savefig(inplot('inertia_after2100.pdf'))


clp()
from paradice.utils import shift

inertia_list = np.linspace(4,18,4)
fig, axs = plt.subplots(3, len(inertia_list), figsize=(16, 10),sharey='row')
for icol, inertia in enumerate(inertia_list):
    solutions_10inertia = df[np.isclose(df[o.o_min_peakabatrate_lab],inertia,rtol=1e-1)].sort_values(o.o_max_rel2c_lab).groupby('miu').first()
    for irow, (x, p) in enumerate(zip([mdps,mtime], prop_list)):
        run = dc[x].run(solutions_10inertia.loc[x][colsmiu[x]])
        run.MIU.loc[:2200].plot(ax=axs[0,icol], legend=False, **p)
        y = run.TATM.loc[:2200]
        y.plot(ax=axs[2, icol], legend=False, **p)
        rel2c = (y < 2).all(0).sum()
        # rel2c = (y<2).all(0).sum()
        axs[2,icol].annotate(f'Rel2C({x}) = {rel2c}',
                         xy=(0, 1), xycoords='axes fraction',
                         xytext=(10, -10*(irow*2+1)), textcoords='offset pixels',
                         horizontalalignment='left',
                         verticalalignment='top')

        #for i, y in run.MIU.loc[:2200].T.iterrows():
        #    ax.plot(y.index, y.values, **p)
        m = dc[x]._mlist[1]
        pd.DataFrame(100*np.array(np.abs(m.MIU[1:]-shift(m.MIU[1:],1))[1:])/m.tstep, index=run.MIU.index[1:]).plot(ax=axs[1,icol], legend=False,**p)
        axs[1,icol].axhline(inertia, color='k', ls='--')
    #axs[1,icol].set_ylim([0,20])
    #axs[0,icol].set_ylim([0,1.3])
fig.tight_layout()
    fig.savefig(inplot(f'inertia{int(inertia):02d}.pdf'))
clp()
inplot('.')
m.MIU1.shape

solution_fields = ['variables', 'objectives', 'problem']
Solution = namedtuple('Solution', solution_fields)

# Check differences between original and rerun dataset
ds = v.load_dataset(inrootdir('sandbox', 'u1w100doeclim_mrbfXdX3_i1p100_nfe4000000c4000000_objgreg4_s1_seed0001_rerun_objall.nc'))
a=ds.to_dataframe()
b = v.load_pareto_mpi(inrootdir('sandbox', 'u1w100doeclim_mrbfXdX3_i1p100_nfe4000000c4000000_objgreg4_s1_seed0001_runtime.csv'))
b = b.loc[4000000]
c=pd.concat([a,b], keys=['new','old'], names=['run','idsol'])
d=c.stack().reset_index()

g = sb.FacetGrid(data=d, col='level_2', col_wrap=5, col_order=o.oset2labs['all'], hue='run',
                 sharex=False, sharey=False)
g.map(sb.distplot, 0)
g.add_legend()

e=c.unstack('run')[o.o_min_npvdamcost_lab];abs(e['new']-e['old']).sort_values(ascending=False)


oset = 'jack6'
o2invert = o.get_mpi_col2invert()[oset]
colsobjs = o.oset2labs[oset]
p = h.args2dice(f'-o {oset}').asproblem()

i2f = lambda i: f'u1w100doeclim_mtime_i1p100_nfe4000000c4000000_objgreg4_s{i}_seed000{i}_rerun_objall.nc'

def f2normdf(f='u1w100doeclim_mtime_i1p100_nfe4000000c4000000_objgreg4_s3_seed0003_rerun_objall.nc'):
    df = v.load_dataset(inrootdir('sandbox', f)).to_dataframe()
    df.loc[:,o2invert] = (-1)*df.loc[:,o2invert]
    return df

a = f2normdf(i2f(1))
refset = a.copy()
vrefset = refset[colsobjs].values
for f in [i2f(i) for i in [2,3]]:
    a = f2normdf(f)
    for x, y in tqdm(a.iterrows(),total=a.shape[0]):
        idx_dominated = np.all(y[colsobjs].values<vrefset, axis=1)
        idx_dominating = np.all(y[colsobjs].values>vrefset, axis=1)
        if np.any(idx_dominated):
            refset = refset[~idx_dominated]
            vrefset = refset[colsobjs].values
        if not np.any(idx_dominating):
            refset.loc[x] = y
            vrefset = refset[colsobjs].values

aplot



refset[refset[o.o_min_peakabatrate_lab]<=4][colsobjs].describe()
colsx = v.get_xcols(a)

a[a[o.o_min_peakabatrate_lab]<=4][o.oset2labs['greg4']].describe()


reference_set = pl.EpsilonBoxArchive(o.oset2epss[oset])
# Invert max objectives
a[colsobjs].describe()
a[colsobjs].describe()



b
    reference_set.add(Solution(variables=y[colsx].values, objectives=y[colsobjs].values, problem=p))


miu = 'rbfXdX3'
orel2c = o.o_max_rel2c_lab
dfx3 = ds.sel(miu=miu).to_dataframe().dropna(1, how='all').dropna(0).sort_values(
        [o.o_max_rel2c_lab, o.o_min_npvmitcost_lab, o.o_min_npvdamcost_lab, o.o_max_util_bge_lab],
        ascending=[False, True, True, False])
dc = h.args2dice(f'{q.miu2arg(miu)} -o all -u 1 -w 100 -e 2200')
colsmiu = [f'dv{i}' for i in range(len(dc.get_bounds()))]

fig, ax = plt.subplots(1,1)
for x, y in tqdm(dfx3.iterrows(), total=dfx3.shape[0]):
    dfx3.loc[x,o.o_min_peakabatrate] = dc.run_and_ret_objs(y[colsmiu])[o.o_min_peakabatrate]

mpl.get_backend()
