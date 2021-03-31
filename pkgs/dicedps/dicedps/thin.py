import sys
import dicedps.viz_dps_policy as v
import dicedps.objectives as o

if __name__ == '__main__':
    fname = sys.argv[1]
    df = v.load_pareto_mpi(fname)

    dfthin = df.sort_values(o.o_min_npvmitcost_lab).groupby(o.o_max_rel2c_lab, as_index=False).first()
    fthin = f'{fname[:-4]}_thinned.csv'
    v.save_pareto(dfthin[df.columns], fthin)

    print(f'Pareto front thinned from {df.shape[0]} to {dfthin.shape[0]} lines')
