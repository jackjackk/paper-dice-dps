import io
import os
import pkgutil

import sys
from enum import Enum, auto
from glob import glob
from pathlib import Path

from paradicedoeclim.dicedoeclim import DiceDoeclim2

from dicedps.objectives import oset2vout
from paradice.dice import Dice, ECS, DiceBase, tfixer, Damages
from paradigm import Data, MODE_OPT, Time, MODE_SIM, MultiModel, partial
import numpy as np
import logging
import pandas as pd
import copy

from dicedps.qsub_dice import get_parsed_args
from dicedps.environ import indatabrick

from . import uncertainties as u
from .dpsrules import MiuRBFController, args2dpsclass, signal2bounds, MiuTemporalController, MiuController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dice_helper')


class DiceClimate(Enum):
    DOECLIM = 'doeclim'
    DICE = 'dice'


def get_uncertain_dicesim(dps_class=MiuRBFController,
                          climate=DiceClimate.DOECLIM,
                          climt0=False,
                          seed=1,
                          ulist=None,
                          endyear=2200,
                          climcalib=None,
                          damfunc=Damages.NORDHAUS,
                          control_kwargs=None,
                          **kwargs):
    """Build a DICE simulator with SOWs setup.

    Parameters
    ----------
    dps_class : MiuController, optional
        If None, use standard DICE temporal MIU variables.
        If `MiuController` object, create a `MultiModel` which alternate
         between DICE and the controller (by default use a `MiuRBFController`).
    climate : DiceClimate, optional
        Climate module to use (default: `DiceClimate.DOECLIM`).
    climt0 : bool, optional
        Include temperature in the initial year in the MCMC sampling (default: False)
    seed : int
        Seed to sample uncertain quantities
    ulist : list, optional
        If None, perform a Latin-Hypercube sampling of the climate sensitivity parameter
         of the specified climate module, with 10 samples (default).
        If list of `u.UncertainParam`, sample from given list.
    endyear : int, optional
        End year of DICE time horizon (default: 2200).
    climcalib : str or function
        If str, passed to `u.get_sows_setup_mcmc` to specify ".nc" calibration filename.
        If function, passed to `u.get_sows_setup` to compute climate setup `dict` given 't2co' values.
        If None (default), passed to `u.get_sows_setup` but `ulist` cannot have 't2co' (otherwise
         no clue on how to calibrate 'kappa' and 'alpha')
    damfunc : DamageFunction, optional
        Set given damage function (default: `DamageFunction.NORDHAUS`).
    control_kwargs : dict, optional
        Extra arguments passed to `dps_class` init or as `control_kwargs` in `DiceDoeclim2` init.
    kwargs
        Extra arguments passed to `DiceBase` init or as `dice_kwargs` in `DiceDoeclim2` init.

    Returns
    -------
    DiceBase
        With no `dps_class` and DICE climate.
    MultiModel
        With `dps_class` and DICE climate.
    DiceDoeclim2
        With `dps_class` and DOECLIM climate.

    """
    # Sample SOWs
    import random
    random.seed(seed)

    # Default uncertainty
    if ulist is None:
        if climate == 'dice':
            uinst = u.ClimateSensitivityUncertainty()
        elif climate == 'doeclim':
            uinst = u.DoeclimClimateSensitivityUncertainty()
        else:
            assert False
        ulist = [u.UncertainParam(uinst, u.sample_latin, 10)]

    if climate == 'doeclim':
        assert climcalib is not None

    # Create SOW-parameters mapping
    if isinstance(climcalib, str):
        if '_fgiss_' in climcalib:
            fsource = 0  #'giss'
        elif '_furban_' in climcalib:
            fsource = 1  #'urban'
        else:
            assert False, f'no supported forcing source found in {climcalib}'
        #forc_datafile = pkgutil.get_data(__package__, 'data/forcings.csv')
        #ret = pd.read_csv(io.BytesIO(forc_datafile), index_col=[0, 1]).xs(fsource,0,'source')
        startyear = 1880
        kwargs_sows_setup = u.get_sows_setup_mcmc(climcalib, nsow=ulist[0].nsow, inct0=climt0)
        kwargs_sows_setup['setup']['forcing_source'] = fsource
    else:
        startyear = 1900
        kwargs_sows_setup = u.get_sows_setup(ulist, climcalib=climcalib)

    kwargs_sows_setup['setup']['damfunc'] = damfunc
    try:
        kwargs_sows_setup['setup'].update(kwargs.pop('setup'))
    except:
        pass

    # Build DICE
    if control_kwargs is None:
        control_kwargs = {}
    dice_args = kwargs
    dice_args_plus_sows = dice_args.copy()
    dice_args_plus_sows.update(kwargs_sows_setup)
    if dps_class is None:
        assert climate=='dice'
        dice_args_plus_sows['sow_setup']['Emission control rate GHGs'] = 0
        dc = DiceBase(mode=MODE_SIM, vin=['MIU'], endyear=endyear, **dice_args_plus_sows)
    else:
        if climate=='dice':
            simdice = DiceBase(mode=MODE_SIM, endyear=endyear, **dice_args_plus_sows)
            simdice.set_inbound('MIU').set_outbound('TATM')
            controller = dps_class(simdice, **control_kwargs)
            dc = MultiModel(controller, simdice)
        elif climate=='doeclim':
            dc = DiceDoeclim2(dps_class, startyear=startyear, endyear=endyear,
                              dice_kwargs=dice_args, control_kwargs=control_kwargs, **kwargs_sows_setup)
    return dc


def get_median_dice(**kwargs):
    d = get_uncertain_dicesim(dps_class=None, mode=MODE_OPT, ulist=[], **kwargs)
    return d


def get_median_dice_bau(**kwargs):
    return get_median_dice(endyear=2500, **kwargs).set_bau().solve()


def get_median_dice_opt(**kwargs):
    return get_median_dice(endyear=2500, **kwargs).solve()

def args2ulist(args):
    nsow = int(args.sows)
    if nsow > 1:
        sampler = u.sample_latin
    else:
        sampler = u.sample_3
    if args.nunc == '1':
        uname = None
    else:
        uname = args.nunc
    return [u.UncertainParam(u.DoeclimClimateSensitivityUncertainty(name=uname), sampler, nsow)]


def args2climcalib(climcalib):
    _args2climcalib = {
        'k18': u.get_kappa_2018,
        'ka18': u.get_kappa_alpha_2018,
        'ka16': u.get_kappa_alpha_2016,
        'c16': u.get_kappa_alpha_cdice2016,
        #'low': 'brick_fgiss_TgissOcheng_schylek_o4',
        #'med': 'brick_fgiss_Tgiss_slognorm_o4',
        #'high': 'brick_fgiss_Tgiss_scauchy_o10',
        #'hig': 'brick_fgiss_Tgiss_scauchy_o10',
       'low': indatabrick('brick_mcmc_fgiss_TgissOgour_schylek_t18802011_z18801900_o4_h50_n20000000_t1000_b5.nc'),
       'med': indatabrick('brick_mcmc_fgiss_TgissOgour_spaleosens_t18802011_z18801900_o4_h50_n20000000_t1000_b5.nc'),
       'high': indatabrick('brick_mcmc_fgiss_TgissOgour_scauchy_t18802011_z18801900_o4_h50_n20000000_t1000_b5.nc'),
    }
    try:
        iclimcalib = int(climcalib)
        # See damages3.py
        _listcalibfiles = ['brick_fgiss_TgissOcheng_schylek_o4',
       'brick_fgiss_TgissOcheng_schylek_o10',
       'brick_fgiss_Tgiss_schylek_o4',
       'brick_fgiss_ThadcrutOcheng_schylek_o4',
       'brick_fgiss_Thadcrut_schylek_o4', 'brick_fgiss_Tgiss_schylek_o10',
       'brick_furban_Tgiss_schylek_o10',
       'brick_furban_TgissOcheng_schylek_o10',
       'brick_furban_TgissOcheng_schylek_o4',
       'brick_fgiss_Thadcrut_schylek_o10',
       'brick_furban_TgissOcheng_scauchy_o4',
       'brick_furban_ThadcrutOcheng_scauchy_o4',
       'brick_furban_Tgiss_scauchy_o4',
       'brick_furban_TgissOcheng_scauchy_o10',
       'brick_furban_Thadcrut_scauchy_o4',
       'brick_furban_Tgiss_slognorm_o4',
       'brick_furban_Thadcrut_slognorm_o4',
       'brick_furban_ThadcrutOcheng_scauchy_o10',
       'brick_fgiss_TgissOcheng_scauchy_o4',
       'brick_furban_Tgiss_slognorm_o10',
       'brick_fgiss_ThadcrutOcheng_scauchy_o4',
       'brick_furban_Tgiss_scauchy_o10',
       'brick_fgiss_TgissOcheng_scauchy_o10',
       'brick_fgiss_Tgiss_slognorm_o4',
       'brick_furban_Thadcrut_slognorm_o10',
       'brick_fgiss_Tgiss_scauchy_o4',
       'brick_furban_Thadcrut_scauchy_o10',
       'brick_fgiss_Thadcrut_slognorm_o4',
       'brick_fgiss_Thadcrut_scauchy_o4',
       'brick_fgiss_Tgiss_slognorm_o10',
       'brick_fgiss_Thadcrut_slognorm_o10',
       'brick_fgiss_Thadcrut_scauchy_o10',
       'brick_fgiss_Tgiss_scauchy_o10']
        ret = _listcalibfiles[iclimcalib]
    except ValueError as e:
        ret = _args2climcalib.get(climcalib, climcalib)
    logger.info(f'Using calibration profile "{ret}"')
    return ret


def args2con(args):
    _args2con = {
        'inertmax': ['PEAKABATRATE_4PCTYR_MAX'],
        'inert95q': ['PEAKABATRATE_4PCTYR_95Q'],
        'none': None,
    }
    return _args2con[args.con]


def args2dice(args, **kwargs):
    if isinstance(args, str):
        args = get_parsed_args(args)
    print(args)
    # TODO: add support for args.nunc, before: ulist=u.ulist3[:args.nunc]
    dbau_path = os.path.join(args.outpath, 'dice_bau.dat')
    if args.reset or (not os.path.exists(dbau_path)):
        logger.info('Running BAU')
        dbau = DiceBase(mode=MODE_OPT, endyear=2500).set_bau().solve()
        dbau.save(dbau_path[:-4])
    else:
        dbau = Data.load(dbau_path)
    if args.miu == 'time':
        logger.warning('Forcing inertmax con')
        args.con = 'inertmax'
    dc = get_uncertain_dicesim(dps_class=args2dpsclass(args),
                               bau=dbau,
                               rsav=dbau,
                               ulist=args2ulist(args),
                               climt0=args.climt0,
                               climcalib=args2climcalib(args.climcalib),
                               damfunc=args.damfunc,
                               seed=args.iseed,
                               endyear=args.endyear,
                               vout=oset2vout[args.obj],
                               con=args2con(args),
                               climate=args.climate,
                               **kwargs)
    dc.asproblem()
    return dc


def check_signal_bounds(args):
    args2 = copy.copy(args)
    args2.miu = 'time'
    dicemin = args2dice(args2)
    dicemax = args2dice(args2)
    miumin = np.zeros(len(dicemin.MIU) - 1)
    miumax = 1.2 * np.ones_like(miumin)
    runmin = dicemin.run(miumin)
    runmax = dicemax.run(miumax)
    for sname, svar, delta in zip(['T', 'dT', 'G', 'B'], ['TATM', 'TATM', 'YGROSS', 'yearccs'], [1, 50, 0]):
        y = pd.concat([getattr(runmin, svar), getattr(runmax, svar)], axis=1, keys=['miu0', 'miu1'])
        if sname == 'dT':
            y = (y-y.shift()).dropna()
        y = y.T.stack()
        bnds = signal2bounds[sname]
        x0 = (y.min()-bnds[0])/(bnds[1]-bnds[0])
        assert x0 >= 0
        x1 = (y.max()-bnds[0])/(bnds[1]-bnds[0])
        assert x1 <= 1
        logger.info(f'Signal {sname} range: {x0:.2f} - {x1:.2f}')
    return True


if __name__ == '__main__':
    if sys.argv[1] == 'bau':
        dbau = DiceBase(mode=MODE_OPT, endyear=2500).set_bau().solve()
        dbau.save('dice_bau')
