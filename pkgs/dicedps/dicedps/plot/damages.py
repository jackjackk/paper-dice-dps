from collections import defaultdict

from dicedps.plot.common import *

os.chdir(os.path.join(os.environ['HOME'], 'working/dicedps/sandbox'))
df = v.load_pareto('*greg4b*_c*merged.csv', revert_max_objs=False, objlabs=False)
pd.DataFrame({i:{'a':i} for i in range(100)}).T.iloc[99:100]
df = pd.concat({miu:dforig.loc[miu].loc[con] for miu, con in zip(miulist,['none','inertmax'])}, axis=0)

fig, ax = plt.subplots(1, 1)
for miu in df.index.levels[1]:
    sb.distplot(df.xs(miu,0,'miulab')[o.o_max_rel2c],ax=ax,label=miu)
ax.legend()

df.xs(miu,0,'miulab')[o.o_max_rel2c].describe()

simdps = get_sim2plot(mdps)
plot_var_cmap(simtime, ['MIU'], sol1pct)
plot_var_cmap(simdps, ['MIU'], sol1pct)


os.chdir('figures')

mit=0.5
mit2maxdiff = {}
# How much does TEMP dist move w/ 2 strategies?
for i, mit in tqdm(enumerate(np.arange(0.4,0.5,0.1))):
    sol1pct = df[np.round(df[o.o_min_cbgemitcost], 1) == mit]
    if len(sol1pct)<1:
        continue
    fig, ax = plt.subplots(1, 1)
    t = {}
    dt = {}
    for s in [simtime, simdps]:
        s.dc.run_and_ret_objs(v.get_x(sol1pct.loc[s.miu].iloc[0]))
        tdf = pd.DataFrame(np.array(s.dc._mlist[3].temp[1:,:]), index=s.dc._mlist[3].year[1:], columns=range(100))
        t[s.miu] = tdf.loc[2100]
        dt[s.miu] = (tdf-(tdf.rolling(30).mean())).loc[2100]
        sb.distplot(t[s.miu], ax=ax, label=s.miu)
    ax.legend()
    ax.set_xlim([0,5])
    ax.set_ylim([0,1])
    mit2maxdiff[mit] = (t[mtime]-t[mdps]).max()
    fig.savefig(f'temp2100_{i:04d}.png', dpi=200)
clp()

s.get('TATM').loc[2100]


md=simtime.dc._doeclim
simtime.dc.run(np.zeros(47))

simtime.dc.TATM.T.describe()




pd.concat({
    'nordhaus': m.damfraceq.f(m,1),

}

)




#ax.plot(m.TATM[1], (m.a1 * m.TATM[1]) + (m.a2 * pow(m.TATM[1], m.a3)), label='Nordhaus poly')
m.a1, m.a2, m.a3
import matplotlib.animation as ani
im_ani = ani.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                   blit=True)
