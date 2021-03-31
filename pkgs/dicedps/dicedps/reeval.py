from dicedps.plot.common import *
import sys


def calc_reeval_given_dist(uname):
    ds = xr.open_dataset('last.nc', engine='netcdf4')

    dc = {}
    df = {}
    colsmiu = {}
    for miu in m2:
        df[miu] = ds.sel(miu=miu).to_dataframe().dropna(1, how='all').dropna(0).sort_values([o.o_max_rel2c_lab,o.o_min_npvmitcost_lab,o.o_min_npvdamcost_lab,o.o_max_util_bge_lab], ascending=[False,True,True,False])
    for miu in [mtime, mdps]:
        dc[miu] = h.args2dice(f'{q.miu2arg(miu)} -o greg4 -u 1 -w 100 -e 2200 -A')
        colsmiu[miu] = [f'dv{i}' for i in range(len(dc[miu].get_bounds()))]



    def get_lowest_mitcost_for_given_rel2c(x, y, col=o.o_min_npvmitcost_lab):
        return y[np.isclose(y[o.o_max_rel2c_lab], x, atol=1e-2)].iloc[0].loc[col]

    rel2cvals = {miu:df[miu][o.o_max_rel2c_lab].unique() for miu in m2}
    solbyrel = {}
    for miu in [mtime,mdps]:
        solbyrel[miu] = pd.DataFrame({x:get_lowest_mitcost_for_given_rel2c(x, df[miu], col=slice(None)) for x in rel2cvals[miu]}).T

    results = {}
    blacklist = set()
    for miu in m2:
        dc = h.args2dice(f'-w 100 -e 2200 {q.miu2arg(miu)} -r 4 -o greg4 -u {uname} -A')
        csmin = np.min(dc._mlist[3].t2co)
        print(csmin)
        if csmin<0.3:
            blacklist.add(uname)
            continue
        for x, sol in tqdm(solbyrel[miu].iterrows(), total=solbyrel[miu].shape[0]):
            results[(miu, uname, sol[o.o_max_rel2c_lab], sol[o.o_min_npvmitcost_lab])] = dc.run_and_ret_objs(sol[colsmiu[miu]])
            break
    if len(results)>0:
        dfresults = pd.DataFrame(results).T
        dfresults.index.rename(['miu', 'dist', 'rel2c', 'mitcost'], inplace=True)
        dfresults.loc[:,['MAX_REL2C','MAX_UTIL_BGE','MIN_NPVDAMCOST','MIN_NPVMITCOST']].to_csv(f'reeval_{uname}.csv')
    else:
        with open(f'reeval_{uname}.csv','w') as f:
            f.write('miu,dist,rel2c,mitcost,MAX_REL2C,MAX_UTIL_BGE,MIN_NPVDAMCOST,MIN_NPVMITCOST\n')


if __name__ == '__main__':
    calc_reeval_given_dist(sys.argv[1])

    """
    for miu in [mtime,mdps]:
        for mu in tqdm(np.linspace(1,5,11)):
            for sigma in tqdm(np.linspace(0.05,0.4,9)):
                try:
                except:
                    pass
    """
    #xr.Dataset.from_dataframe(dfresults).to_netcdf(inrootdir('sandbox/reeval.nc'), format='NETCDF4')
