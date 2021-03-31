import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pylab as plt
plt.interactive(True)
import seaborn as sb

import rhodium as rh
import platypus as pl
from tqdm import tqdm

from importlib import reload

import logging
import os
import sys
logging.basicConfig(level=logging.DEBUG)

from . import dpsrules as r
from . import uncertainties as u
from . import objectives as o
from . import dice_helper as h
from . import viz_dps_policy as v
from . import qsub_dice as q

from paradigm import Time
from paradice.dice import Dice, DiceBase
from paradigm import MODE_OPT, MODE_SIM, Data
from paradigm.viz import pplot

import numpy as np
import pandas as pd
pd.set_option('display.precision', 5)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pandas.plotting as pdp
from functools import partial
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
import xarray as xr
import scipy.stats as ss
import sympy as sy
import sympy.stats as syst
import scipy.integrate as sci

#from j3 import J3
#import pygmo as pg

from glob import glob

clp = lambda: plt.close('all')

from sklearn.externals import joblib

from dicedps.environ import *


def save_data(df, name):
    '''
    Save dataframe to be retrieved later w/ load_data and name.

    :param df: dataframe to save
    :param name: label for later recall
    :return: None
    '''
    fname = inoutput('dicedps', f'{name}.dat')
    try:
        df.to_parquet(fname)
    except:
        joblib.dump(df, fname)


def load_data(*vlist):
    '''
    Load datasets from cache dir.

    :param vlist: datasets names
    :return: DataFrame (if len(vlist)==1) or dict w/ loaded datasets
    '''
    data = {}
    for vcurr in vlist:
        try:
            data[vcurr] = pd.read_parquet(inoutput('dicedps', f'{vcurr}.dat'))
        except:
            data[vcurr] = joblib.load(inoutput('dicedps', f'{vcurr}.dat'))
    if len(vlist) == 1:
        return data[vcurr]
    return data