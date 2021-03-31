from tqdm import tqdm
import argparse
import dicedps.dice_helper as h
import dicedps.viz_dps_policy as v
import dicedps.objectives as o
import dicedps.qsub_dice as q

import os
import numpy as np
import pandas as pd
import xarray as xr

#fname = 'u1w100doeclim_mrbfXdX3_i1p100_nfe4000000c4000000_objgreg4_s1_seed0001_runtime.csv'
#obj = 'all'
#%cd ~/working/dicedps/sandbox


if __name__ == '__main__':
    parser = q.get_parser()

    # Solver
    parser.add_argument('-I', '--input', dest='input', type=str, action='store',
                        help='input runtime file to reevaluate')
    parser.add_argument('-P', '--output', dest='output', type=str, action='store',
                        help='output runtime file to reevaluate')
    parser.add_argument('-l', '--nlines', dest='nlines', type=int, action='store',
                        help='number of lines to process (0 = all)', default=0)
    parser.add_argument('-L', '--nsegment', dest='nsegment', type=int, action='store',
                        help='segment number to process (0=all, 1 - 100)', default=0)
    parser.add_argument('-N', '--ncpus', dest='ncpus', type=int, action='store',
                        default=1, help='Number of cpus')

    args = parser.parse_args()
    fname = args.input
    #fname = os.path.join(os.environ['HOME'], 'working', 'dicedps', 'sandbox', 'u1w1000doeclim_mrbfXdX_i1p40_nfe4000000_objjack5_s0_seed0000_merged_thinned.csv')
    dict_args_matching_output = q.name2dargs(args.output, save_miulab=False)
    obj = dict_args_matching_output['obj']

    dfpar, dfpar_meta = v.load_pareto_mpi(fname, objlabs=False, metadata=True)

    # Consider only last NFE
    df = dfpar.xs(dfpar.index.levels[0][-1],0,'nfe')
    if args.nlines != 0:
        df = df.head(args.nlines)
    idx = df.index
    n = df.shape[0]
    dfx = df[v.get_xcols(dfpar)]
    if args.nsegment > 0:
        m = ((n-1)//99)
        assert m>0
        if args.nsegment < 100:
            dfx = dfx.iloc[(m*(args.nsegment-1)):(m*(args.nsegment))]
        else:
            dfx = dfx.iloc[(m*(args.nsegment-1)):]

    # Build DICE simulator
    #dcdargs = q.name2dargs(fname)
    #for k, v in dcdargs.items():
    #    oldv = getattr(args, k)
    #    if oldv != v:
    #        print(f'--{k}: replacing {oldv} with {v}')
    #        setattr(args, k, v)
    #orig_obj = dcdargs['obj']
    #assert orig_obj != obj, f'New obj "{obj}" should be different from original obj "{orig_obj}"'
    #dcdargs['obj'] = obj
    str_args_matching_output = q.dargs2sargs(dict_args_matching_output)
    print(str_args_matching_output)
    
    dc = h.args2dice(str_args_matching_output)

    # Run
    if args.ncpus == 1:
        a = {}
        for i, x in tqdm(dfx.iterrows(), total=dfx.shape[0]):
            idx, objs = dc.run_and_ret_single((i, x))
            a[idx] = objs
    else:
        a = dc.run_parallel(dfx.iterrows(), ncpus=args.ncpus)
    b=pd.DataFrame(a).T[o.oset2vout[obj]]

    # Write out file
    df2write = pd.merge(dfx, b, left_index=True, right_index=True).rename(o.obj2lab, axis=1)
    fout = args.output
    fbase, fext = os.path.splitext(fout)
    fext = fext[1:]
    if args.nsegment > 0:
        if not os.path.exists(fbase):
            try:
                os.mkdir(fbase)
            except:
                pass
        fout = os.path.join(fbase, f'{fbase}_{args.nsegment:04d}.{fext}')
    if os.path.exists(fout):
        os.remove(fout)
    print(f'Writing {fout}...')
    if fext == 'nc':
        ds = xr.Dataset.from_dataframe(df2write)
        ds.to_netcdf(fout, format='NETCDF4')
    elif fext == 'csv':
        v.save_pareto(df2write, fout)
    else:
        raise Exception(f'Format {fext} not recognized')

    # Check objectives
    """orig_obj_list = list(o.oset2vout[orig_obj])
    orig_obj_list.remove('MAX_UTIL_BGE') # TODO
    dforig2check = df[orig_obj_list]
    dfrerun2check = b.loc[idx,orig_obj_list]
    assert np.allclose(dfrerun2check, dforig2check)"""



#ds=xr.open_dataset('u1w100doeclim_mrbfXdX3_i1p100_nfe4000000c4000000_objgreg4_s1_seed0001_rerun_objall.nc')
#c=pd.DataFrame({'new':b.loc[idx,'MAX_UTIL_BGE'],'old':dforig['MAX_UTIL_BGE']})
#import matplotlib.pylab as plt
#c.plot()
#plt.show()

