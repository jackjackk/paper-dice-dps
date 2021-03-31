import numpy as np
from scipy import stats
from paradice.dice import ECS, Dice, DiceBase
from paradigm import MODE_SIM, Time, partial
from paradoeclim.utils import get_kappa_2018, get_kappa_alpha_2018, get_kappa_alpha_2016, get_kappa_alpha_cdice2016, get_kappa_alpha_borg2018
import random
from collections import namedtuple
import scipy.stats as ss
import os
import pkgutil
import pandas as pd
import tqdm
import scipy.integrate as sci
import io
import xarray as xr

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uncertainties')

instantiate_if_needed = lambda x: x() if isinstance(x, type) else x

"""From Rhodium"""


UncertainParam = namedtuple('UncertainParam', ['cls','sampler','nsow'])


class NamedObject(object):
    """Object with a name."""

    def __init__(self, name):
        super(NamedObject, self).__init__()
        self.name = name

from abc import ABCMeta, abstractmethod


class Uncertainty(NamedObject):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        super(Uncertainty, self).__init__(name)

    @abstractmethod
    def levels(self, nlevels):
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def ppf(self, x):
        raise NotImplementedError("method not implemented")


class UniformUncertainty(Uncertainty):

    def __init__(self, name, min_value, max_value, **kwargs):
        super(UniformUncertainty, self).__init__(name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def levels(self, nlevels):
        d = (self.max_value - self.min_value) / nlevels
        result = []

        for i in range(nlevels):
            result.append(self.min_value + random.uniform(i * d, (i + 1) * d))

        return result

    def ppf(self, x):
        return self.min_value + x * (self.max_value - self.min_value)

class NormalUncertainty(Uncertainty):

    def __init__(self, name, mean, stdev, **kwargs):
        super(NormalUncertainty, self).__init__(name)
        self.mean = float(mean)
        self.stdev = float(stdev)

    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, 0.0, 1.0).levels(nlevels)
        return stats.norm.ppf(ulevels, self.mean, self.stdev)

    def ppf(self, x):
        return stats.norm.ppf(x, self.mean, self.stdev)


class LogNormalUncertainty(Uncertainty):

    def __init__(self, name, mu, sigma, **kwargs):
        super(LogNormalUncertainty, self).__init__(name)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, 0.0, 1.0).levels(nlevels)
        return self.mu * stats.lognorm.ppf(ulevels, self.sigma)

    def ppf(self, x):
        return self.mu * stats.lognorm.ppf(x, self.sigma)



class AugmentedUncertainty(object):

    def ppf_levels(self, nlevels):
        x0 = 1/nlevels
        x1 = 1-x0
        xlist = np.linspace(x0, x1, nlevels)
        return [self.ppf(x) for x in xlist]


    def param_name(self):
        return self.name


    def param_desc(self):
        return self.__doc__


class ClimateSensitivityRV(ss.rv_continuous):

    def __init__(self, name='olson_informPrior', pkgdir=None):
        super().__init__()
        self.name = name
        fobj = pkgutil.get_data(__package__, os.path.join('data', 'parsed_pdfs', f'{name}.txt'))
        self.data = pd.read_csv(io.BytesIO(fobj), sep=' ', header=None)
        ncols = self.data.shape[1]
        collabels = ['cs', 'pdf', 'cdf']
        self.data.columns = collabels[:ncols]
        assert np.min(self.data.cdf)>=0.
        if ncols == 2:
            self.data.sort_values(by='cs', ascending=True, inplace=True)
            self.data['cdf'] = self.data['pdf'].mul(np.nan)
            self.data['cs1'] = self.data['cs'].shift()
            self.data['cdf2'] = self.data['cdf'].copy()
            for x, y in tqdm(self.data.iterrows(), total=len(self.data)):
                y['cdf2'] = sci.quad(self.pdf, a=y['cs1'], b=y['cs'])[0]
                if np.isnan(y['cdf2']):
                    y['cdf2'] = 0.
                # y['cdf'] = sci.quad(self.pdf, a=self.data.cs.min(), b=y['cs'])[0]
            self.data['cdf'] = self.data['cdf2'].cumsum()
            self.data.drop(['cdf2', 'cs1'], axis=1, inplace=True)
            lastpoint = self.data['cdf'].iloc[-1]
            assert np.isclose(lastpoint, 1., atol=1e-3)
            self.data['cdf'] = self.data['cdf'].mul((1 - 2 * 1e-6) / lastpoint).add(1e-6)
            print(name, [self.data['cdf'].iloc[i] for i in [0, -1]])
            self.data['cdf'] = np.minimum(self.data['cdf'], 1. - 1e-6)
            assert self.data['cdf'].iloc[-1] < 1.
            fname = os.path.join(pkgdir, 'data', 'parsed_pdf', f'{name}.txt')
            self.data[collabels].to_csv(fname, sep=' ', header=False, index=False, float_format='%.18f')

    @staticmethod
    def available_distributions():
        return ['aldrin_a','aldrin_ecs2','aldrin_f','aldrin_h','aldrin_k','hargreaves_ayako','hargreaves_bayes','hargreaves_grl','kohler','lewis','lewis_rev','libardoni_extend','libardoni_grl','olson_informPrior','olson_unifPrior','otto_2000s','otto_avg','palaeosens','schmittner_all','schmittner_land','schmittner_ocean','tomassini','tomassini_expertPrior']


    def _pdf(self, x):
        return np.maximum(0, np.interp(x, self.data['cs'], self.data['pdf']))


    def _ppf(self, q):
        return np.maximum(0, np.interp(q, self.data['cdf'], self.data['cs']))


    def plot(self, ax=None, **kws):
        import matplotlib.pylab as plt
        if ax is None:
            fig, ax = plt.subplots(1,1)
        return ax.plot(self.data['cs'], self.data['pdf'], **kws)


class ClimateSensitivityUncertainty(LogNormalUncertainty, AugmentedUncertainty):
    __doc__ = 'Equilibrium temp impact (oC per doubling CO2)'

    def __init__(self):
        super().__init__('t2xco2', np.exp(ECS.mu), ECS.sigma)




class DoeclimClimateSensitivityUncertainty(Uncertainty, AugmentedUncertainty):
    __doc__ = 'Climate sensitivity (K)'


    def __init__(self, name=None, mu=None, sigma=None):
        if name is None:
            if mu is None:
                mu = np.exp(ECS.mu)
            if sigma is None:
                sigma = ECS.sigma
            self.rv = ss.lognorm(s=sigma, scale=mu)
        else:
            self.rv = ClimateSensitivityRV(name)
        super().__init__('t2co')


    def levels(self, nlevels):
        ulevels = UniformUncertainty(self.name, 0.0, 1.0).levels(nlevels)
        return self.ppf(ulevels)


    def ppf(self, x):
        return self.rv.ppf(x)


class OutputGrowthUncertainty(NormalUncertainty, AugmentedUncertainty):
    __doc__ = 'Initial growth rate for TFP per 5 years'

    def __init__(self):
        ga0_ppf01 = 0.0085 #
        ga0_ppf09 = 0.255
        norm_icdf01 = stats.norm.ppf(0.1)
        norm_icdf09 = stats.norm.ppf(0.9)
        # Derive normal distribution parameters from percentiles
        std_ga0 = (ga0_ppf09 - ga0_ppf01) / (norm_icdf09 - norm_icdf01)
        mean_ga0 = ga0_ppf01 - std_ga0 * norm_icdf01

        super().__init__('ga0', mean=mean_ga0, stdev=std_ga0*0.9)


class BackstopAvailabilityUncertainty(UniformUncertainty, AugmentedUncertainty):
    __doc__ = 'Initial year at which abat can be greater than 100%'

    def __init__(self, startyear=2100, endyear=2200):
        super().__init__('yearccs', startyear, endyear)

    def levels(self, nlevels):
        return np.arange(self.min_value, self.max_value+10, 10)[:nlevels]
        #ret = super().levels(nlevels)
        #return [(int(x) - int(x) % 5) for x in ret]


ulist3 = [ClimateSensitivityUncertainty,
          OutputGrowthUncertainty,
          BackstopAvailabilityUncertainty]
ulist1 = ulist3[:1]
ulist2 = ulist3[:2]

ulist1b = [DoeclimClimateSensitivityUncertainty]

ulist_all = ulist3 + ulist1b

def _sample_multiplexer_multisampler(ulist, samplerlist):
    """
    Stack multiple uncertainties samples into vectors to be used as setup.

    """
    ret_setup = {}
    if len(ulist) == 1:
        u0 = ulist[0]
        ret_setup[u0.name] = samplerlist[0](u0)
    elif len(ulist) == 3:
        u0, u1, u2 = [instantiate_if_needed(u) for u in ulist]
        ul0, ul1, ul2 = [sampler1dim(u) for sampler1dim, u in zip(samplerlist, [u0,u1,u2])]
        nul0, nul1, nul2 = [len(ul) for ul in [ul0, ul1, ul2]]
        ret_setup[u0.name] = np.repeat(ul0, nul1 * nul2)
        ret_setup[u1.name] = np.tile(np.repeat(ul1, nul0), nul2)
        ret_setup[u2.name] = np.tile(ul2, nul0 * nul1)
    return ret_setup

def _sample_multiplexer_1sampler(ulist, sampler1dim):
    """
    Stack multiple uncertainties samples into vectors to be used as setup.

    :param ulist: List of Uncertainty classes/objects
    :param sampler1dim: function to apply to each u in ulist to get realizations
    :return: dict of { u.name : [ stacked_realizations ] }
    """
    samplerlist = [sampler1dim]*len(ulist)
    return _sample_multiplexer_multisampler(ulist, sampler1dim)


def uncertainparam_sampler_multiplexer(ulist):
    """
    Stack multiple uncertainties samples into vectors to be used as setup.

    :param ulist: List of Uncertainty classes/objects
    :param sampler1dim: function to apply to each u in ulist to get realizations
    :return: dict of { u.name : [ stacked_realizations ] }
    """
    return _sample_multiplexer_multisampler([unc.cls for unc in ulist], [partial(unc.sampler, nsow=unc.nsow) for unc in ulist])


def sample_latin(u, nsow):
    return u.levels(nsow)


def sample_combinatorial_general(ulist, nsowdict):
    return _sample_multiplexer_1sampler(ulist, lambda u: u.levels(nsowdict[u.__class__]))


def sample_combinatorial(ulist, nsow1dim):
    return sample_combinatorial_general(ulist, {u.__class__: nsow1dim for u in ulist})


def sample_median(u):
    return [u.ppf(0.5),]

def sample_3(u, nsow=1):
    return [3.,]

sample_latin1000 = partial(sample_latin, nsow=1000)
sample_latin100 = partial(sample_latin, nsow=100)
sample_latin10 = partial(sample_latin, nsow=10)
sample_latin3 = partial(sample_latin, nsow=3)
sample_latin5 = partial(sample_latin, nsow=5)
sample_latin1 = sample_median


def sample_extremes(u, eps=1e-6):
    assert (eps>0) and (eps<1.00001e-2)
    try:
        levels = [u.min_value, u.max_value]
    except:
        levels = [u.ppf(eps), u.ppf(1-eps)]
    return levels


def ncbrick2pandas(ncfile, columns=['t2co', 'kappa', 'alpha', 'temp0']):
    #if ncfile is None:
    #    ncfile = os.path.join(os.environ['HOME'], 'working', 'learn-brick', 'results', 'doeclim_mcmc_fgreg_sinf_e2015_t1929_o4_n10000000_b5_t1000_n1.nc')
    ncbr = xr.open_dataset(ncfile)

    # Extract parameters names
    a = ncbr['parnames'].to_dataframe()
    map_names = a['parnames'].unstack().apply(lambda x: ''.join([str(i, 'utf-8') for i in x]), axis=1)

    # Extract data
    df = ncbr['BRICK_parameters'].to_dataframe()
    # sb.distplot(df.xs(1,0,'n.parameters'))
    if columns is None:
        colslice = slice(None)
    else:
        colslice = slice(len(columns))
    ppdata = df['BRICK_parameters'].unstack('n.parameters').iloc[:, colslice]
    ppdata.columns = map_names.iloc[colslice].values
    if columns is not None:
        ppdata.columns = columns
    return ppdata


def get_sows_setup_mcmc(ncfile, nsow=100, inct0=False, seed=1):
    if not os.path.exists(ncfile):
        logger.info('{ncfile} does not exist, trying within package')
        ncfile_data = pkgutil.get_data(__package__, f'data/{ncfile}.nc')
        ncfile = io.BytesIO(ncfile_data)
    df = ncbrick2pandas(ncfile)
    if nsow<df.shape[0]:
        if nsow <= 100:
            dfsub = df.sort_values('t2co').iloc[::(df.shape[0]//nsow)]
        else:
            state = np.random.get_state()
            np.random.seed(seed)
            dfsub = df.iloc[np.random.randint(low=0, high=df.shape[0], size=nsow)]
            np.random.set_state(state)
    else:
        logger.warning(f'Asked for {nsow} SOWs, but only {df.shape[0]} available.')
        dfsub = df
    setup = {v: col.values for v,col in dfsub[['t2co', 'kappa', 'alpha']].items()}
    sow_setup = {}
    sow_setup['Climate sensitivity (K)'] = nsow
    sow_setup['Ocean heat diffusivity (cm^2 s^-1)'] = nsow
    sow_setup['Aerosol forcing scaling factor'] = nsow
    if inct0:
        setup['temp0'] = dfsub['temp0'].values
        sow_setup['Initial temperature (K)'] = nsow
    default_sow = nsow
    return {'setup': setup,
            'sow_setup': sow_setup,
            'default_sow': default_sow}


def get_sows_setup(ulist, climcalib=None):
    setup = uncertainparam_sampler_multiplexer(ulist)
    name2doc = {u.cls.name: u.cls.__doc__ for u in ulist}
    sow_setup = {}
    default_sow = 0
    clim_setup = {}
    for k, v in setup.items():
        nsow = len(v)
        sow_setup[name2doc[k]] = nsow
        default_sow = max(nsow, default_sow)
        if k=='t2co':
            assert 't2xco2' not in setup
            clim_setup = climcalib(setup['t2co'])
            if 'kappa' in clim_setup:
                sow_setup['Ocean heat diffusivity (cm^2 s^-1)'] = nsow
            if 'alpha' in clim_setup:
                sow_setup['Aerosol forcing scaling factor'] = nsow
    setup.update(clim_setup)
    return {'setup': setup,
            'sow_setup': sow_setup,
            'default_sow': default_sow}


""""
    # Make sure uncertain parameters are SOW-dependant
    for u in ulist:
		#TODO
        if u in ulist:
            nsow = len(list(setup[uinst.name]))
        else:
            nsow = 0
            setup[uinst.name] = uinst.ppf(0.5)
        default_sow = max(default_sow, nsow)
        sow_setup[u.__doc__] = nsow
        if u == OutputGrowthUncertainty:
            sow_setup['Decline rate of TFP per 5 years'] = nsow
            ga0 = setup['ga0']
            setup['dela'] = 1/(37*5)*(np.log(np.abs(ga0)) - np.log(1e-3) + np.log(1+np.sign(ga0)*1e-3))
        elif u == DoeclimClimateSensitivityUncertainty:
            sow_setup['Ocean heat diffusivity (cm^2 s^-1)'] = nsow
            if use_calib_alpha:
                sow_setup['Aerosol forcing scaling factor'] = nsow
                clim_setup = get_kappa_alpha(setup['t2co'])
            else:
                clim_setup = get_kappa(setup['t2co'])
            setup.update(clim_setup)
            if nsow > 0:
                disable_t2xco2 = True
        # # Limit use of backstop to 100 years, initially introduced for feasibility reasons, then dropped
        # elif u == BackstopAvailabilityUncertainty:
        #    yearccs = dice_setup['yearccs']
        #    dice_setup['yearnoccs'] = np.array(yearccs+100+5)
        #    try:
        #        dice_sow_setup['Year at which abat stops being potentially greater than 100%'] = len(yearccs)
        #    except:
        #        pass
    if disable_t2xco2 and ('t2xco2' in setup):
        setup['t2xco2'] *= np.nan
    return {'setup': setup,
            'sow_setup': sow_setup,
            'default_sow': default_sow}
    #logger.info('Uncertainties\n' + str(pd.DataFrame(dice_setup).describe()))
    """


get_sows_setup_dicedoeclim = partial(get_sows_setup, ulist_all=ulist1b)
