import re
from collections import OrderedDict, namedtuple
import xarray as xr
import matplotlib as mpl
import matplotlib.pylab as plt
import pandas as pd
import glob
import os
import numpy as np
from paradice.dice import Dice, ECS
from paradigm import Data, MODE_OPT, Time, MODE_SIM, MultiModel
import rhodium as rh
import itertools as it

import logging

from tqdm import tqdm

import dicedps.objectives as do
from dicedps.qsub_dice import fname2index
from dicedps.dice_helper import get_uncertain_dicesim
from dicedps.dpsrules import MiuRBFController, MiuPolyController, miulab2nvars
from dicedps.environ import *
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils')


def _load_pareto_serial(filename):
    y = pd.read_csv(filename, comment='/')  # .set_index(['NFE', 'index'])  # [['obj0', 'obj1']]
    assert len(y) > 0
    nfe = y.NFE.max()
    y = y[y.NFE == nfe]
    return y


def process_last_lines(ncols,
                       last_lines,
                       objs,
                       obj_labeller,
                       nfe,
                       rets,
                       last_props,
                       rets2):
    """Parse a set of lines and extracted meta-data (corresponding to one Borg iteration)
    into entries of two dictionaries, using the number of function evaluations as key.

    Parameters
    ----------
    ncols : int
        Number of columns.
    last_lines : list of str
        List of lines corresponding to one Borg iteration.
    objs : list of str
        List of objectives
    obj_labeller
        Function that transform (e.g. prettify) an objective label.
    nfe : int
        Number of function evaluations for the iteration.
    rets : dict
        Dictionary to update with a (nfe, pd.DataFrame parsed from `last_lines`)
        key-value pair.
    last_props : dict
        Parsed meta-data.
    rets2 : dict
        Dictionary to update with a (nfe, pd.Series from `last_props`)
        key-value pair.

    """
    nobjs = len(objs)
    nrows = len(last_lines)
    data = np.zeros((nrows, ncols))
    for i, l in enumerate(last_lines):
        try:
            data[i, :] = [float(x) for x in l.split()]
        except:
            #badcount += 1
            print(f'Bad line {i}, {len(l.split())} fields instead of {data.shape[1]}')
    ret = pd.DataFrame(data, columns=[f'dv{j}' for j in range(ncols - nobjs)] + \
                                     [obj_labeller(o) for o in objs])
    rets[nfe] = ret
    last_props['NSOL'] = len(ret)
    rets2[nfe] = pd.Series(last_props)


def load_pareto_mpi(filename: str,
                    objlabs: bool=True,
                    keep_only_last: bool=True,
                    revert_max_objs: bool=True, ndvs: int=None,
                    metadata: bool=False):
    """Load a single MPI Borg runtime csv into a DataFrame.

    Parameters
    ----------
    filename : str
        Runtime filename to parse.
    objlabs : bool, optional
        Use objective labels instead of raw string codes (default).
    keep_only_last : bool
        Keep only results of the last Borg iteration, if more than one are reported (default).
    revert_max_objs : bool, optional
        Compensate for Borg inversion of sign of max objectives,
        making sure sign of objectives matches original meaning (default).
        This means inverting the sign of maximization objectives.
    ndvs : int, optional
        Number of decision variables. If None, inferred from 'miulab' index key (default), inferred from filename.
    metadata : bool, optional
        Return also extracted metadata.

    Returns
    -------
    ret : pd.DataFrame
        Rows = solutions, columns = variables and objectives
    ret2 : pd.DataFrame, optional
        Metadata

    """
    logger.info(f'Processing "{filename}"')
    brecording = False
    last_lines = []
    last_props = {}
    pat = re.compile(r'//NFE=(\d+)')
    pat2 = re.compile(r'//(\w+)=(.+)$')
    args = fname2index(filename)
    obj = args['obj']
    obj_labeller = do.labeller(objlabs)
    objs = do.oset2vout[obj]
    nobjs = len(objs)
    if ndvs is None:
        ndvs = miulab2nvars(args.miulab)
    nfe = np.nan
    data = None
    rets = {}
    rets2 = {}
    counter = 0
    badcount = 0
    with open(filename, 'r') as f:
        for line in f:  # read each line
            if line[0] in ['/', '#']:
                brecording = False
                if (nfe is not None) and (len(last_lines)>0):
                    process_last_lines(ndvs + nobjs, last_lines, objs, obj_labeller, nfe, rets, last_props, rets2)
                    last_lines = []
                    last_props = {}
                if line[0] == '/':
                    try:
                        nfe = int(pat.match(line).groups()[0])
                    except:
                        mlist = pat2.match(line).groups()
                        last_props[mlist[0]] = float(mlist[1])
            else:
                if not brecording:
                    brecording = True
                    counter = 0
                last_lines.append(line)
                counter += 1
    if len(last_lines)>0:
        process_last_lines(ndvs + nobjs, last_lines, objs, obj_labeller, nfe, rets, last_props, rets2)
    if (len(rets)>1) and keep_only_last:
        rets = {nfe: rets[nfe]}
        rets2 = {nfe: rets2[nfe]}
    ret = pd.concat(rets, names=['nfe','idsol'])
    ret2 = pd.concat(rets2, names=['nfe','variable'])
    if revert_max_objs:
        logger.info('Inverting "max" objectives')
        for col in do.get_mpi_col2invert(objlabs).get(obj, []):
            ret[col] = -ret[col]
    # sort columns
    if metadata:
        return ret, ret2
    return ret


def load_pareto(filelist, **kwargs):
    """Load a set of MPI Borg runtime csv files.

    Read files into DataFrames.
    Concatenate DataFrames using keys derived from filenames via `fname2index`.
    Drop index levels with just one value.

    Parameters
    ----------
    filelist
        Pareto csv files to load, specified as a list of paths or as a glob pattern string.
    kwargs
        Arguments to pass to load_pareto_mpi

    Returns
    -------
    DataFrame
        Rows = solutions, columns = variables and objectives.

    """
    dfs = {}
    df2s = {}
    logger.info('Reading files...')
    if isinstance(filelist, str):
        filelist = glob.glob(filelist)
    assert len(filelist)>0, f'No file found under "{os.getcwd()}" matching "{filelist}"'
    for filename in tqdm(filelist):  # os.path.join('examples', 'dice', f'*{smatch}*.csv')):
        idx = fname2index(filename)
        if idx['procs'] == 1:
            # TODO
            assert False
            df = _load_pareto_serial(filename)
        else:
            df, df2 = load_pareto_mpi(filename, ndvs=None, metadata=True, **kwargs)
        dfs[tuple(idx.values)] = df
        df2s[tuple(idx.values)] = df2
    logger.info('Concatenating...')
    dfbig = pd.concat(dfs, names=idx.index.values.tolist())
    levels2drop = np.where(pd.DataFrame(list(dfs.keys()), columns=idx.index).apply(lambda x: len(x.unique()))==1)[0].tolist()
    logger.info(f'Dropping levels {idx.index.values[levels2drop]}')
    dfbig.index = dfbig.index.droplevel(levels2drop)
    # logger.info('Turning into dataset...')
    # ret = xr.Dataset.from_dataframe(dfbig)
    logger.info('Done')
    return dfbig


def save_pareto(df: pd.DataFrame, fout: str):
    """
    Append dataframe to csv.

    Parameters
    ----------
    df
    fout

    Returns
    -------

    """
    logger.info(f'Writing {fout}...')
    df.to_csv(fout, sep=' ', index=False, header=False)
    with open(fout, "a") as myfile:
        myfile.write("#")


def convert_runtime_csv_into_dataset(csvlist, out='last.nc'):
    pfs = load_pareto(csvlist)
    try:
        pfs.index = pfs.index.droplevel('procs')
    except:
        pass
    ds = xr.Dataset.from_dataframe(pfs)
    logger.info('Writing NetCDF file...')
    ds.to_netcdf(out, format='NETCDF4')
    return ds


def load_dataset(f='last.nc'):
    return xr.open_dataset(f)


flastres = os.path.join(os.environ['HOME'], 'CloudStation','psu','projects','dice-dps','last_results.feather')


def update_last_feather(saveto=None, **kwargs):
    ds = load_pareto(intestdir('batch', '*500000'), **kwargs)
    if saveto is None:
        saveto = flastres
    else:
        saveto = intestdir(saveto)
    logger.info(f'Saving to "{saveto}"')
    ds.to_feather(saveto)
    return ds

def load_last_feather(loadfrom=None):
    if loadfrom is None:
        loadfrom = flastres
    else:
        loadfrom = intestdir(loadfrom)
    return (pd.read_feather(loadfrom)
            .set_index(['unc','control','borg','obj','seed','idsol'])
            .sort_index())


def get_simulator(dps_class=MiuRBFController):
    return get_uncertain_dicesim(dps_class)


def get_t_dt_bound():
    dc = get_simulator()
    ds = load_last_feather()
    dsr = ds[(ds.miu == 'dpsr/serial') & (ds.obj == 'simple2')].dropna(1)
    nx = len(list(c for c in dsr.columns if c[:2]=='dv'))
    xcols = [f'dv{i:d}' for i in range(nx)]
    tmax=dtmax=0
    tmin=dtmin=100
    tall = []
    dtall = []
    for idx, row in tqdm(dsr.iterrows()):
        dc.run(row[xcols])
        t = dc._mlist[1].TATM[1:]
        tm1 = np.roll(t, shift=1, axis=0)
        tm1[0,:] = np.nan
        dt = (t - tm1)[1:]
        tall.append(t)
        #t2min = np.insert(t, t.shape[0], tmin, axis=1)
        #t2max = np.insert(t, t.shape[0], tmax, axis=1)
        #tmax, tmin = t2max.max(), t2min.min()
        #dt2min = np.insert(dt, t.shape[0], dtmin, axis=1)
        #dt2max = np.insert(dt, t.shape[0], dtmax, axis=1)
        #dtmax, dtmin = dt2max.max(), dt2min.min()
        dtall.append(dt)
    t = np.concatenate(tall, axis=1)
    dt = np.concatenate(dtall, axis=1)
    return tmin, tmax, dtmin, dtmax

def get_t_dt_bound_approx():
    return 0.5, 6, -0.05, 0.4

def _columns_or_index(df):
    if isinstance(df, pd.DataFrame):
        clist = df.columns
    else:
        clist = df.index
    return clist

def get_xcols(df):
    clist = _columns_or_index(df)
    nx = len(list(c for c in clist if c[:2]=='dv'))
    xcols = [f'dv{i:d}' for i in range(nx)]
    return xcols

def get_x(df):
    """
    Return array or list of arrays with input values
    taken from a Pareto dataframe `df`.

    Parameters
    ----------
    df
        `pd.DataFrame` with rows = solutions, columns = variables and objectives,
        or `pd.Series` with index = columns and variables.

    Returns
    -------
    list
        If `df` is `pd.Series`, list of extracted variables.
    list of list
        If `df` is `pd.DataFrame, list of list of extracted variables, one for each row.

    """
    #if isinstance(df, pd.DataFrame):
    #    logger.warning(f'Picking 1 out of {len(df)} rows')
    #    df = df.iloc[0]
    return df[get_xcols(df)].dropna().values.tolist()

def get_ocols(df):
    xcols = get_xcols(df)
    clist = _columns_or_index(df)
    olist = [col for col in clist if col not in xcols]
    olist_set = set(olist)
    for k,v in do.oset2labs.items():
        if set(v) == olist_set:
            return v
    logger.warning('No oset found, returning unordered objective list')
    return olist

def get_o(df):
    return df[get_ocols(df)]

def get_sols_by_rel2c(miu='dpsr/serial', obj='simple2', orel2c='obj0'):
    ds = load_last_feather()
    dsr = ds[(ds.miu == miu) & (ds.obj == obj)].dropna(1)
    xcols = get_xcols(dsr)
    dsr['REL2C'] = dsr[orel2c].round(0)
    ret = dsr.groupby('REL2C').median()
    return ret


def calc_rbf(t, dt, x):
    n = 3
    miu = np.maximum(
        np.minimum(
            1.2,
            sum([x[0+j] * np.exp(-((t - x[n*2+j]) / x[n+j]) ** 2 -
                                 ((dt - x[n*4+j]) / x[n*3+j]) ** 2)
                 for j in range(0, n)])),
        0)
    return miu


def calc_quad(t, dt, x):
    miu = np.maximum(
        np.minimum(
            1.2,
            (x[0] * t +
             x[1] * (t) ** 2 +
             x[2] * (dt) +
             x[3] * (dt) ** 2 +
             x[4] * t * dt)),
        0)
    return miu


def scatter2d(df, **kwargs):
    df = df.reset_index()
    df.as_dataframe = lambda: df
    """
    try:
        c = kwargs['c']
        if df[c].dtype == np.dtype('O'):
            df[c] = df[c].astype('category')
            leg_labels = df[c].categories.tolist()
            leg_handles = 
            df[c] = df[c].cat.codes
    except:
        pass
    """
    RhodiumModel = namedtuple('Model', ['responses'])
    m = RhodiumModel(responses={col: None for col in df.columns if col[:4]=='obj_'})
    rh.scatter2d(m, df, **kwargs)


if __name__ == '__main__':
    convert_runtime_csv_into_dataset(sys.argv[1:], out='last.nc')





