from collections import defaultdict

from dicedps.plot.common import *

inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','paper-dice-dps','meetings','20180601-keller-group-update','figures',*x)

df2degy = load_merged('greg4d')


simtime = get_sim2plot(mtime)
simdps = get_sim2plot(mdps)
m=simtime.dc._dice
md=simtime.dc._doeclim

amiu, out = get_sol_by_mitcost(df2degy.loc[mtime], -0.8, relmax=True, retout=True, atol=1e-3, sortby=o.o_min_mean2degyears_lab)
simtime.dc.run(amiu)
amiu, out = get_sol_by_mitcost(df2degy.loc[mdps], -0.8, relmax=True, retout=True, atol=1e-3, sortby=o.o_min_mean2degyears_lab)
simdps.dc.run(amiu)

dfburke=pd.read_csv(os.path.join(os.environ['HOME'],'Downloads','burke_damages2018.csv'), header=None)
dfgoes=pd.read_csv(os.path.join(os.environ['HOME'],'Downloads','goes_damages2011.csv'))


from matplotlib.gridspec import GridSpec
def plot_grid():
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(2, 2)
    ax_dam = plt.subplot(gs[0, 0])
    ax_temp = plt.subplot(gs[1, 0])
    ax_box = plt.subplot(gs[:,1])
    return fig, np.array([ax_dam, ax_temp, ax_box])



fig, axs = plot_grid()

# Plot temp dists
ax=axs[0]
for s in [simtime,simdps]:
    sb.distplot(s.get('TATM').loc[2100],ax=ax,label=miu2lab[s.miu])
ax.legend()
ax.set_ylabel('PDF')
ax.set_xlabel('Temperature in 2100 [K]')
ax.set_xlim([0.5,4])


ax=axs[1]
goesfunc = lambda m, a1, a2: 100*np.tanh(a1*np.power(m.TATM[18]/3., 2)+a2*np.power((m.TATM[18]-m.TATM[17])/5/0.35,4))
tolfunc2 = lambda x: -(3.99*x-1.82*np.power(x,2))
tolfunc6 = lambda x: -(0.348*x-0.0109*np.power(x,6))

import numpy.polynomial.polynomial as poly
coefs = poly.polyfit(dfburke[0], -dfburke[1], 2)
ffit = poly.polyval(m.TATM[18], coefs)

weitfunc = lambda x: 100*(1-1./(1.+ 0.0028388*np.power(x,2)+0.0000050703*np.power(x,6.754)))

damlab2func = {
 'Nordhaus': (lambda m: 100*m.damfraceq.f(m,18)),
 'Tol(2)': (lambda m: tolfunc2(m.TATM[18])),
 'Tol(6)': (lambda m: tolfunc6(m.TATM[18])),
 'Burke': (lambda m: poly.polyval(m.TATM[18], coefs)),
 'Goes(Max)': (lambda m: goesfunc(m, 0.17,8.3e-19)),
 'Weitzman': (lambda m: weitfunc(m.TATM[18])),
 #'GoesB': (lambda: goesfunc(0.02,0.0028)),
}

m = simtime.dc._dice
for (k, v), p in zip(damlab2func.items(), prop_list):
    ax.plot(m.TATM[18], v(m), **p)
    ax.annotate(s=k, xy=(m.TATM[18][-1],v(m)[-1]), va='top', ha='center')

ax.set_xlim([0.5,4])
ax.set_xlabel('Temperature in 2100 [K]')
ax.set_ylabel('GDP loss in 2100 [%]')


ax = axs[2]

#clp()

damdict = defaultdict(dict)
damdf = {}
for damlab, damfunc in damlab2func.items():
    for s in [simtime, simdps]:
        m = s.dc._dice
        damdict[damlab][miu2lab[s.miu]] = pd.Series(np.array(damfunc(m)), index=range(100))
    damdf[damlab] = pd.concat(damdict[damlab])

dfalldam=pd.concat(damdf).reset_index()
sb.boxplot(x='level_0',order=['Nordhaus','Tol(2)','Tol(6)','Goes(Max)','Burke'],y=0,hue='level_1',hue_order=['Non-adaptive','Adaptive'], data=dfalldam,ax=ax,showmeans=True,meanprops=dict(markeredgecolor='black',markerfacecolor='black'))
ax.set_xlabel('Damage function')
ax.set_ylabel('GDP loss in 2100 [%]')
fig.tight_layout()
fig.savefig(inplot('fig_damages.png'),dpi=200)



"""
dftemp=pd.DataFrame(np.array(md.temp[1:,:]), index=md.year[1:], columns=range(100))
t2dt = pd.concat([dftemp, dftemp-(dftemp.rolling(30).mean())], keys=['T','dT']).unstack(0).loc[2015:].stack(0)
#fig, ax = plt.subplots(1,1)
#t2dt[np.isclose(t2dt,2,atol=1e-1)][1].max()
tchanges_min = np.zeros_like(m.TATM[18])
tchanges_max = np.zeros_like(m.TATM[18])
for i, t in enumerate(np.array(m.TATM[18])):
    tchanges_min[i] = t2dt[np.isclose(t2dt['T'], t, atol=1e-1)]['dT'].min()
    tchanges_max[i] = t2dt[np.isclose(t2dt['T'], t, atol=1e-1)]['dT'].max()

#ax.plot(m.TATM[18], tchanges_min)
#ax.plot(m.TATM[18], tchanges_max)
#m.TATM[18] = np.linspace(1,6,100)

#for a1, a2 in zip(dfgoes['alpha1'], dfgoes['alpha2']):
#    hgoes = ax.plot(m.TATM[18], goesfunc(a1, a2), label='Goes', **prop_list[2])
#        #ax.annotate(f'{a1},{a2}',xy=(m.TATM[18][-1],ys[-1][-1]))
#sb.regplot(dfburke[0],-dfburke[1], lowess=False, order=2, ax=ax, label='Burke 2018', **prop_list[1])
#hnord = ax.plot(m.TATM[18], 100*m.damfraceq.f(m,18), label='Nordhaus', lw=2, color='k')
#htol = ax.plot(m.TATM[18], tolfunc(m.TATM[18]), label='Tol', **prop_list[0])

#hburke = ax.plot(m.TATM[18], ffit, **prop_list[3])

#ax.legend([hnord[0], htol[0], hgoes[0], hburke[0]], ['Nordhaus', 'Tol', 'Goes', 'Burke'])

"""