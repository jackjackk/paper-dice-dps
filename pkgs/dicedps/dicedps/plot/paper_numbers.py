from dicedps.plot.common import *
import string

df = load_merged(orient='min')

dmaxutil = {}
for m in [mdps,mtime]:
    dmaxutil[m] = df.loc[m].sort_values(o.o_max_util_bge_lab).iloc[0][o.oset2labs[last_oset]]

olab2dec = {
    o.o_min_mean2degyears_lab: 0,
}

with open(inpaper('..','paper-numbers.org'),'w') as f:
    for k, v in pd.concat(dmaxutil).unstack().mean().items():
        f.write(f'#+MACRO: max-util-{k.replace("_","-").lower()} {v:.{olab2dec.get(k,2)}f}\n')

    for k, v in df.groupby('miulab')[o.o_min_cbgemitcost_lab].agg(['min','max']).mean().items():
        f.write(f'#+MACRO: {k}-mitcost {v:.1f}\n')
