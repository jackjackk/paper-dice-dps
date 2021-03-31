import operator
from collections import defaultdict

from dicedps.plot.common import *

#df = v.load_pareto_mpi('u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#v.save_pareto(df, 'u1w1000doeclim_mtime_i1p100_nfe4000000_objgreg4b_cinertmax_s3_seed0003_last.csv')
#df[v.get_ocols(df)]

df = load_merged(orient='min')
o2decoffset = {o.o_min_mean2degyears_lab:2}
dfround = df.round({olab: ndecs-o2decoffset.get(olab, 1) for olab, ndecs in zip(o.oset2labs[last_oset], o.oset2decs(last_oset))})

objlist = o.oset2labs[last_oset]
dfn, _ = get_scaled_df(dfround[objlist])

sb.set_context('paper')
fig, ax = plt.subplots(1,1,figsize=(5,6))
for objcurr, p in zip(objlist, prop_list):
    for miucurr, ls in zip([mtime,mdps], ['-','--']):
        dfncurr2 = dfn.loc[miucurr].sort_values(objcurr)
        dfncurr = dfncurr2.drop_duplicates()
        print(f'{len(dfncurr2)} -> {len(dfncurr)}')
        dfcurr = dfround.loc[miucurr].loc[dfncurr.index]
        if miucurr == mtime:
            k = {'color':'0.5'}
        else:
            k = p
        dfncurr[dfncurr[objcurr]<(dfncurr[objcurr].iloc[0]+1e-2)].mean().T.plot(ax=ax, alpha=0.5, legend=False, **k)




dfcurr.head()
dfncurr.head()

dfntime = dfntime.groupby(dfntime[o.o_min_mean2degyears_lab].round(2)).mean().reindex(np.arange(0,1,0.01)).interpolate()
dfndps = dfn.loc[mdps].sort_values(o.o_min_mean2degyears_lab)
dfndps = dfndps.groupby(dfndps[o.o_min_mean2degyears_lab].round(2)).mean().reindex(np.arange(0,1,0.01)).interpolate()
