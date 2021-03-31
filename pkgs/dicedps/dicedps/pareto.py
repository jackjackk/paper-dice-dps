import sys
from glob import glob
import os
import dicedps.viz_dps_policy as v
import dicedps.objectives as o
import dicedps.qsub_dice as q
import numpy as np
from tqdm import tqdm


def thin(x, oset='jack5', declist=[0, 2, 2, 2, 1]):
    ret = x.round(dict(zip(o.oset2labs[oset], declist)))
    rows_before = ret.shape[0]
    ret.drop_duplicates(inplace=True)
    rows_removed = rows_before-ret.shape[0]
    if rows_removed>0:
        print(f'Removed {rows_removed} duplicate solutions')
    return ret


if __name__ == '__main__':
    flist = sys.argv[1:]
    flist = glob(os.path.join(os.environ['HOME'],
                              'working','dicedps','sandbox','u1w1000*XdX*objjack5*last.csv'))

    dflist = []
    nsol_table = {}
    run_feats = None  # series of flags common to all runs (must differ at most for seed)
    refset = None  # dataframe with resulting reference set
    vrefset = None  # corresponding np.array
    oset = None  # label of the objective set used
    colsobjs = None  # columns representing objective values
    for f in flist:
        dfpar, dfpar_meta = v.load_pareto_mpi(f, objlabs=True, revert_max_objs=False)
        df = dfpar.xs(dfpar.index.levels[0][-1],0,'nfe')
        curr_run_feats = q.fname2index(f)
        nsol_table[curr_run_feats['seed']] = df.shape[0]
        df = thin(df)
        nsol_table[f'{curr_run_feats["seed"]}thinned'] = df.shape[0]
        if run_feats is None:
            # First pareto front in the set, init data structures
            run_feats = curr_run_feats
            oset = run_feats['obj']
            colsobjs = o.oset2labs[oset]
            refset = df
            vrefset = refset[colsobjs].values
        else:
            # After first pareto front
            # Check for homogeneity with first
            assert np.all(run_feats==curr_run_feats), f'{flist[0]} and {f} differ'
            # For each solution
            for x, y in tqdm(df.iterrows(), total=df.shape[0]):
                idx_dominated = np.all(y[colsobjs].values < vrefset, axis=1)
                idx_dominating = np.all(y[colsobjs].values > vrefset, axis=1)
                if np.any(idx_dominated):
                    refset = refset[~idx_dominated]
                    vrefset = refset[colsobjs].values
                if not np.any(idx_dominating):
                    refset.loc[x] = y
                    vrefset = refset[colsobjs].values
    nsol_table['merged'] = refset.shape[0]
    print(nsol_table)