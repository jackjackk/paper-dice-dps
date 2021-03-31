from dicedps.plot.common import *

write_macro = write_macro_generator('supp_tempdist')

simdps10k = get_sim2plot(mdps, 10000)
simtime10k = get_sim2plot(mtime, 10000)

# supp plot
from scipy.interpolate import InterpolatedUnivariateSpline
fig2, axs = plt.subplots(3,2, figsize=(w2col, hhalf), sharex=True, sharey='col')
x = np.arange(1, 6.01, 0.01)

df = load_merged(orient='min')
df09mit = df[np.isclose(df[o.o_min_cbgemitcost_lab],0.9,atol=1e-4)]

hpoints = {}
hpoints['A'] = df09mit.loc[mdps].sort_values(o.o_min_mean2degyears_lab).iloc[0]
hpoints['N1'] = df09mit.loc[mtime].sort_values(o.o_min_mean2degyears_lab).iloc[0]

simlist = [simdps10k, simtime10k]
for s, sim in zip(['A','N1'], simlist):
    sim.dc.run(v.get_x(hpoints[s]))


for iyear, year in enumerate([2050,2100,2200]):
    temp2100 = {s: sim.get('TATM').loc[year] for s, sim in zip(['A','N1'], simlist)}
    dftemp2100 = pd.concat(temp2100, names=['miu','sow'])
    a=pd.concat([dftemp2100, dftemp2100.unstack('miu').rank(pct=True).T.stack()], axis=1, keys=['temp','rank'])
    y = {}
    for s, pro in zip(['A','N1'],prop_list):
        b = a.loc[s].drop_duplicates().sort_values(by='temp')
        f = InterpolatedUnivariateSpline(b['temp'].values, b['rank'].values, k=1)
        y[s] = np.minimum(1, np.maximum(0, 1-f(x)))
        axs[iyear,1].plot(x, y[s], label=s, **pro)
        sb.distplot(temp2100[s], label=s, ax=axs[iyear,0], **pro)
        sb.rugplot([temp2100[s].median()], ax=axs[iyear,0], label=f'{s} (Median)', **pro)
    if year == 2100:
        extemp = 2.5
        write_macro('fig02-ex-righttail-temp', f'{extemp:.1f}', True)
        exprob = 100.*pd.DataFrame(y, index=x).reindex([extemp], method='nearest').mean()
        write_macro('fig02-ex-righttail-prob-N1', f'{exprob["N1"]:.0f}')
        write_macro('fig02-ex-righttail-prob-A', f'{exprob["A"]:.0f}')
    for icol, ylab in enumerate(['PDF', 'CDF']):
        ax = axs[iyear,icol]
        ax.legend()
        ax.set_xlabel(lab_temp_year(year))
        ax.set_ylabel(ylab)

savefig4paper(fig2, 'supp_tempdist')
