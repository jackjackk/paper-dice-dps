import io
import os
import pkgutil
from collections import namedtuple
from functools import partial
import pandas as pd
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

# from https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

def get_hist_forc_data():
    forc_datafile = pkgutil.get_data(__package__, 'data/forcings.csv')
    ret = pd.read_csv(io.BytesIO(forc_datafile), sep=',',
                index_col=[0,1])
    return ret


def get_hist_heat_data(m):
    temp_datafile = pkgutil.get_data(__package__, f'data/gouretski_ocean_heat_3000m.txt')
    ret = pd.read_csv(io.BytesIO(temp_datafile), sep=' ', skipinitialspace=True, header=None, skiprows=[0, ], index_col=0)[1]
    return dict(zip(range(1, len(m.t)), ret.reindex(m.year[1:])))


def get_hist_temp_data(m=None, name=None):
    if name is None:
        name = 'hadcrut5'
    _name2file = {
        'giss': 'giss_temp_anomalies_1880_2017.csv',
        'hadcrut4': 'HadCRUT.4.4.0.0.annual_ns_avg.txt',
        'hadcrut5': 'HadCRUT.4.5.0.0.annual_ns_avg.txt',
    }
    temp_datafile = pkgutil.get_data(__package__, f'data/{_name2file[name]}')
    if name == 'giss':
        ret = pd.read_csv(io.BytesIO(temp_datafile), skiprows=1, index_col=0, dtype={'J-D':pd.np.float64})['J-D']
    else:
        ret = pd.read_csv(io.BytesIO(temp_datafile), sep=' ', skipinitialspace=True, index_col=0)['Median']
    if m is not None:
        ret = dict(zip(range(1, len(m.t)), ret.reindex(m.year[1:])))
    return ret


CalibrationSetup = namedtuple('CalibrationSetup', ['name', 'filename', 'params'])

CALIB_OCEANDIFF_AEROSOL_2017 = CalibrationSetup(name='kappa_alpha',
                                           filename='calibration_results_kappa_alpha_garner20170607.txt',
                                           params=['kappa','alpha'])
CALIB_OCEANDIFF_AEROSOL_2016 = CalibrationSetup(name='kappa_alpha',
                                           filename='calibration_results_kappa_alpha_garner20160628.txt',
                                           params=['kappa','alpha'])
CALIB_OCEANDIFF   = CalibrationSetup(name='kappa',
                                           filename='calibration_results_kappa_garner20171027.txt',
                                           params=['kappa',])
CALIB_BORG = CalibrationSetup(name='borg',
                              filename='borg_calib_20180323.csv',
                              params=['kappa','alpha'])


def get_calib_setup(t2co, calib_mode):
    calib_datafile = pkgutil.get_data(__package__, os.path.join('data', calib_mode.filename))
    df = pd.read_csv(io.BytesIO(calib_datafile), header=None, sep=' ',
                names=['t2co',]+calib_mode.params, index_col=False)
    cspline = {x: InterpolatedUnivariateSpline(df['t2co'], df[x], k=1) for x in calib_mode.params}
    return {x: cspline[x](t2co).tolist() for x in calib_mode.params}


get_kappa_alpha_2018 = partial(get_calib_setup, calib_mode=CALIB_OCEANDIFF_AEROSOL_2017)
get_kappa_alpha_2016 = partial(get_calib_setup, calib_mode=CALIB_OCEANDIFF_AEROSOL_2016)
get_kappa_2018 = partial(get_calib_setup, calib_mode=CALIB_OCEANDIFF)
get_kappa_alpha_borg2018 = partial(get_calib_setup, calib_mode=CALIB_BORG)
get_kappa_alpha_cdice2016 = lambda x: {'kappa':0.55+np.array(x)*0, 'alpha':1.98+np.array(x)*0}


