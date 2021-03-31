from collections import defaultdict
from functools import partial
import os
import operator

from paradice.utils import shift
from paradigm import Model, PARAM, VAR, CONSTRAINT, EQUATION, OBJECTIVE, RANGESET, SET, Time, Data, MODE_SIM, MODE_OPT, BRUSH
import datetime
import pyomo.environ as pe
from pyomo.core.base.var import Var, SimpleVar
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.param import Param, SimpleParam
from pyomo.core.base.objective import Objective, maximize, minimize
from paradigm.model import summation, ex
from pyomo.core.base.suffix import Suffix
import numpy as np
import pandas as pd
from math import exp, log
import logging
import sys

logger = logging.getLogger('paradice')

from enum import Enum, IntEnum


class ECS():
    from scipy.stats import lognorm
    sigma = 0.264
    mu = 1.10704
    #mean = 3.13
    #stddev = 0.843
    dist = lognorm(s=sigma, scale=exp(mu))

class TgtType():
    TYPE_NONE, TYPE_ACTIVE = range(2)
    LOWER, UPPER = range(2)
    _, PAR_STARTYEAR, PAR_THRES = range(3)

    @staticmethod
    def limiter(x,u):
        def eqfunc(m, t):
            if m.tgttype[x,u] == TgtType.TYPE_NONE:
                return None
            fromyear, thres = [m.tgtparam[(x,u,i)] for i in range(1,3)]
            if m.year[t] < fromyear:
                return None
            return thres
        return eqfunc



class Ctrl():
    TYPE_FREE, TYPE_FIX_SCALAR, TYPE_FIX_SERIES, TYPE_LOGISTIC, TYPE_EXPONENTIAL, TYPE_CONST, TYPE_EXTERNAL = range(7)
    OP_EQ, OP_LE = range(2)
    eqtype2op = {OP_EQ: operator.eq, OP_LE: operator.le}

    _, PAR_OP, PAR_STARTYEAR, PAR_RATE, PAR_STARTFRAC, PAR_MAXLEVEL = range(6)

    @staticmethod
    def controller(u):
        def eqfunc(m, t):
            if m.ctrltype[u] == Ctrl.TYPE_FREE:
                return None
            if m.ctrltype[u] == Ctrl.TYPE_EXTERNAL:
                return np.nan
            v = getattr(m, u)
            if v[t].fixed:
                return None
            if m.ctrltype[u] == Ctrl.TYPE_FIX_SCALAR:
                return m.ctrlparam[(u, 1)]
            if m.ctrltype[u] == Ctrl.TYPE_FIX_SERIES:
                return m.ctrlparam[(u, t)]
            if m.ctrltype[u] == Ctrl.TYPE_CONST:
                return v[t-1] if t>1 else None
            opnum = m.ctrlparam[(u, 1)]
            if np.isnan(opnum): opnum = Ctrl.OP_EQ
            #m.ctr op = Ctrl.eqtype2op[opnum]
            if m.ctrltype[u] == Ctrl.TYPE_LOGISTIC:
                x1, kspeed, y1, L = [m.ctrlparam[(u,i)] for i in range(2,6)]
                if np.isnan(x1): x1 = 2020
                if np.isnan(y1): y1 = 0.1
                if np.isnan(L): L = 1.
                k = kspeed / L * 4.
                return L / (1 + m.exp(-k * (m.year[t] - (x1 + log(1 / y1 - 1.) / k))))
            if m.ctrltype[u] == Ctrl.TYPE_EXPONENTIAL:
                yearstart, alpha = [m.ctrlparam[(u,i)] for i in range(2,4)]
                if np.isnan(yearstart): yearstart = 2015
                if m.year[t] <= yearstart:
                    return None
                return v[t-1]*(1+alpha)**m.tstep
        return eqfunc



def vbounderwinit(bnds, tfixfunc=None):
    vinit = None
    if callable(bnds):
        def vinitwithlo(m, t):
            lo = bnds(m, t)[0]
            toadd = np.maximum(abs(lo) * 0.01, 0.01)
            return (lo + toadd)

        vinit = vinitwithlo
    else:
        if bnds[0] is not None:
            lo = bnds[0]
            vinit = lambda m,t: lo + np.maximum(abs(lo) * 0.01, 0.01)
    if tfixfunc is not None:
        def vfinal(m,t):
            x = tfixfunc(m,t)
            if not m.isfixed(tfixfunc.varname, t):
                x = vinit(m,t)
            return x
    else:
        vfinal = vinit
    return {'bounds':bnds, 'initialize':vfinal}


def tlevelizer(varname, val):
    def tlevel(m,t):
        if getattr(m, varname)[t].fixed:
            return None
        return val
    return tlevel


def tfixer(varname, valname, cond):
    def tfix(m, t):
        if cond(m, t):
            m.setfixed(varname, t, True)
        return getattr(m, valname)
    tfix.varname = varname
    return tfix


def tfirstfixer(varname, valname, tthres_add=1, tthres_mul=0):
    return tfixer(varname, valname, lambda m,t: t <= (tthres_add+tthres_mul*m.tnum))


def logisticgrowthfixer(what):
    def logisticfix(m, t):
        if getattr(m, what)[t].fixed == True:
            return None
        getattr(m, what)[t].fixed = True
        x1 = 2015
        y1 = getattr(m, '{what}0'.format(what=what.lower())).value
        L = getattr(m, '{what}_max'.format(what=what.lower())).value
        k = getattr(m, '{what}_speed'.format(what=what.lower())).value / L * 4.
        #k = kyr #*m.tstep
        return L/(1+m.exp(-k*(m.year[t]-(x1+log(1/y1-1.)/k))))
    return logisticfix

def logisticfixer(what, startlevel=0.1):
    def logisticfix(m, t):
        if getattr(m, what)[t].fixed == True:
            return None
        getattr(m, what)[t].fixed = True
        x1 = getattr(m, '{what}_start'.format(what=what.lower())).value
        L = getattr(m, '{what}_max'.format(what=what.lower())).value
        k = getattr(m, '{what}_speed'.format(what=what.lower())).value / L * 4.
        y1 = startlevel
        logger.info('k={k},L={L},x1={x1},y1={y1}'.format(k=k,L=L,x1=x1,y1=y1))
        #k = kyr #*m.tstep
        return L/(1+m.exp(-k*(m.year[t]-(x1+log(1/y1-1.)/k))))
    return logisticfix

def exponentialfixer(what, startlevel=0.1):
    def logisticfix(m, t):
        if getattr(m, what)[t].fixed == True:
            return None
        getattr(m, what)[t].fixed = True
        x1 = getattr(m, '{what}_start'.format(what=what.lower()))
        L = getattr(m, '{what}_max'.format(what=what.lower()))
        k = getattr(m, '{what}_speed'.format(what=what.lower())) / L * 4.
        y1 = startlevel
        #k = kyr #*m.tstep
        return L/(1+m.exp(-k*(m.year[t]-(x1+log(1/y1-1.)/k))))
    return logisticfix


def logisticfixer2(what, start, speed, max, startlevel=0.1):
    #x1 = ystart
    #x2 = ystart+yspeed
    #y1 = 0.1
    #y2 = 0.9
    #kyr = yspeed/L*4.
    #k = log(y1/y2*(1-y2)/(1-y1))/(x1-x2)
    #x0 = x1+log(1/y1-1.)/k
    #print(f'LOGISTICFIXER: x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}, k = {k}, x0 = {x0}')
    def logisticfix(m, t):
        if getattr(m, what)[t].fixed == True:
            return None
        getattr(m, what)[t].fixed = True
        x1 = start
        L = max
        k = speed / L * 4.
        y1 = startlevel
        #k = kyr/m.tstep
        return L/(1+m.exp(-k*(m.year[t]-(x1+m.log(1/y1-1.)/k))))
    return logisticfix


def simparval(x):
    try:
        ret = x.value
    except:
        ret = x._default_val
    return ret


def miubounder(m, t):
    bndlo = m.miumin[t] * m.yearccs / m.yearccs
    bndup = m.limmiu0 * (m.year[t] < m.yearccs) \
        + m.limmiu * (m.year[t] >= m.yearccs) * (m.year[t] < m.yearnoccs) \
        + m.limmiu1 * (m.year[t] >= m.yearnoccs)
    return (bndlo, bndup)


class Damages(object):
    TOL2, NORDHAUS, WEITZMAN, GOESMAX, TOL6 = range(5)

    dam2func = [
        lambda m, t: -(3.99 * m.TATM[t] - 1.82 * np.power(m.TATM[t], 2)) / 100.,  # Tol2
        lambda m, t: (m.a1 * m.TATM[t]) + (m.a2 * pow(m.TATM[t], m.a3)),  # Nordhaus
        lambda m, t: (1 - 1. / (1. + 0.0028388 * np.power(m.TATM[t], 2) + 0.0000050703 * np.power(m.TATM[t], 6.754))),  # Weitzman
        lambda m, t: np.tanh(0.17 * np.power(m.TATM[t] / 3., 2)),  # Goes max
        lambda m, t: -(0.348 * m.TATM[t] - 0.0109 * np.power(m.TATM[t], 6)) / 100.,  # Tol6
    ]

    @staticmethod
    def damfrac(m, t):
        idamfunc = int(m.getvalue(m.damfunc))
        damfunc = Damages.dam2func[idamfunc](m, t)
        if idamfunc != Damages.NORDHAUS:
            return m.minimum(0.99, damfunc)
        return damfunc




class DiceBase(Model):

    plot_vlist_single = [['MIU', 'S'],
                         ['E', 'EBASE', 'EABAT', 'EIND'],
                         ['MAT', ],  # 'PPMOCE'],
                         ['YGROSS', 'C', 'I', 'ABATECOST', 'DAMAGES'],
                         ['TATM', ],
                         ['CEMUTOTPER', ],
                         ['MCABATE', 'CPRICE']]
    plot_vlist_multi = ['MIU', 'EIND', 'EABAT', 'MAT', 'TATM',
                        'CPC', 'S', 'DAMAGES', 'ABATECOST', 'MCABATE']


    def __init__(self, mode, endyear=2500, rsav=None, bau=None, **kwargs):
        time = Time(start=2015, end=endyear, tstep=5)
        if bau is not None:
            if isinstance(bau, str):
                bau = Data.load(bau)
            self._bau_miu = np.r_[np.nan, bau.MIU_year.loc[time._range.year].values]
        self._bau = bau

        super().__init__(time=time, mode=mode, **kwargs)

        if rsav is not None:
            self.fix('S', rsav)
            logger.info('Fixing "S" to "{rsav}"'.format(rsav=rsav))
            #self.add_rule('S', tfixer('S', 'optlrsav', (lambda m,t: True)))
        else:
            assert (mode==MODE_OPT) or \
                   ('S' in kwargs.get('vin',[])) or \
                   ('S' in kwargs.get('setup')), \
                   'Either a) provide S via rsav/setup b) add S to vin c) use MODE_OPT'


    def _body(self):
        if self._bau is not None:
            miumin_init = lambda m, t: self._bau_miu[t]
        else:
            miumin_init = lambda m, t: 1e-6
        self.miumin = self.new(PARAM, self.t, doc='Lower bound for MIU', sow=0,
                              initialize=miumin_init)

        # PARAMETERS - SCALAR / EXOGENOUS
        ## Not specified in original DICE
        #self.tnum = Param(doc='Number of time periods', default=100) <- calculated from endyear
        self.cumetree0 = self.new(PARAM, doc='Initial cumulative etree', default=100., sow=0)
        self.e2cume = self.new(PARAM, doc='Emissions conversion factor', default=(12./44.), sow=0)
        self.cca0 = self.new(PARAM, doc='Initial cumulative industrial carbon emissions (GtC)', default=400., sow=0)
        self.yearccs = self.new(PARAM, doc='Initial year at which abat can be greater than 100%', default=2160, sow=0)
        self.yearnoccs = self.new(PARAM, doc='Year at which abat stops being potentially greater than 100%', default=2500, sow=0)
        ## Availability of fossil fuels
        self.fosslim = self.new(PARAM, doc='Maximum cumulative extraction fossil fuels (GtC)', default=6000., sow=0)
        ## If optimal control
        self.ifopt = self.new(PARAM, doc='Indicator where optimized is 1 and base is 0', default=0, sow=0)
        ## Preferences
        self.elasmu = self.new(PARAM, doc='Elasticity of marginal utility of consumption', default=1.45, sow=0)
        self.prstp = self.new(PARAM, doc='Initial rate of social time preference per year', default=.015, sow=0)
        ## Population and technology
        self.gama = self.new(PARAM, doc='Capital elasticity in production function', default=.300, sow=0)
        self.pop0 = self.new(PARAM, doc='Initial world population 2015 (millions)', default=7403, sow=0)
        self.popadj = self.new(PARAM, doc='Growth rate to calibrate to 2050 pop projection', default=0.134, sow=0)
        self.popasym = self.new(PARAM, doc='Asymptotic population (millions)', default=11500, sow=0)
        self.dk = self.new(PARAM, doc='Depreciation rate on capital (per year)', default=.100, sow=0)
        self.q0 = self.new(PARAM, doc='Initial world gross output 2015 (trill 2010 USD)', default=105.5, sow=0)
        self.k0 = self.new(PARAM, doc='Initial capital value 2015 (trill 2010 USD)', default=223., sow=0)
        self.a0 = self.new(PARAM, doc='Initial level of total factor productivity', default=5.115, sow=0)
        self.ga0 = self.new(PARAM, doc='Initial growth rate for TFP per 5 years', default=0.076, sow=0)
        self.dela = self.new(PARAM, doc='Decline rate of TFP per 5 years', default=0.005, sow=0)
        ## Emissions parameters
        self.gsigma1 = self.new(PARAM, doc='Initial growth of sigma (per year)', default=-0.0152, sow=0)
        self.dsig = self.new(PARAM, doc='Decline rate of decarbonization (per period)', default=-0.001, sow=0)
        self.eland0 = self.new(PARAM, doc='Carbon emissions from land 2015 (GtCO2 per year)', default=2.6, sow=0)
        self.deland = self.new(PARAM, doc='Decline rate of land emissions (per period)', default=.115, sow=0)
        self.e0 = self.new(PARAM, doc='Industrial emissions 2015 (GtCO2 per year)', default=35.85, sow=0)
        self.miu0 = self.new(PARAM, doc='Initial emissions control rate for base case 2015', default=.03, sow=0)
        ## Carbon cycle - Initial Conditions
        self.mat0 = self.new(PARAM, doc='Initial Concentration in atmosphere 2015 (GtC)', default=851, sow=0)
        self.mu0 = self.new(PARAM, doc='Initial Concentration in upper strata 2015 (GtC)', default=460, sow=0)
        self.ml0 = self.new(PARAM, doc='Initial Concentration in lower strata 2015 (GtC)', default=1740, sow=0)
        self.mateq = self.new(PARAM, doc='Equilibrium concentration atmosphere  (GtC)', default=588, sow=0)
        self.mueq = self.new(PARAM, doc='Equilibrium concentration in upper strata (GtC)', default=360, sow=0)
        self.mleq = self.new(PARAM, doc='Equilibrium concentration in lower strata (GtC)', default=1720, sow=0)
        ## Carbon cycle - Flow paramaters
        self.b12 = self.new(PARAM, doc='Carbon cycle transition matrix', default=.12, sow=0)
        self.b23 = self.new(PARAM, doc='Carbon cycle transition matrix', default=0.007, sow=0)
        ## Climate self parameters
        self.t2xco2 = self.new(PARAM, doc='Equilibrium temp impact (oC per doubling CO2)', default=3.1)
        self.fex0 = self.new(PARAM, doc='2015 forcings of non-CO2 GHG (Wm-2)', default=0.5, sow=0)
        self.fex1 = self.new(PARAM, doc='2100 forcings of non-CO2 GHG (Wm-2)', default=1.0, sow=0)
        self.tocean0 = self.new(PARAM, doc='Initial lower stratum temp change (C from 1900)', default=.0068, sow=0)
        self.tatm0 = self.new(PARAM, doc='Initial atmospheric temp change (C from 1900)', default=0.85, sow=0)
        self.c1 = self.new(PARAM, doc='Climate equation coefficient for upper level', default=0.1005, sow=0)
        self.c3 = self.new(PARAM, doc='Transfer coefficient upper to lower stratum', default=0.088, sow=0)
        self.c4 = self.new(PARAM, doc='Transfer coefficient for lower level', default=0.025, sow=0)
        self.fco22x = self.new(PARAM, doc='Forcings of equilibrium CO2 doubling (Wm-2)', default=3.6813, sow=0)
        ## Climate damage parameters
        self.a10 = self.new(PARAM, doc='Initial damage intercept', default=0., sow=0)
        self.a1 = self.new(PARAM, doc='Damage intercept', default=0., sow=0)
        self.a2 = self.new(PARAM, doc='Damage quadratic term', default=0.00236, sow=0)
        self.a3 = self.new(PARAM, doc='Damage exponent', default=2.00, sow=0)
        ## Abatement cost
        self.expcost2 = self.new(PARAM, doc='Exponent of control cost function', default=2.6, sow=0)
        self.pback = self.new(PARAM, doc='Cost of backstop 2010$ per tCO2 2015', default=550, sow=0)
        self.gback = self.new(PARAM, doc='Initial cost decline backstop cost per period', default=.025, sow=0)
        self.limmiu0 = self.new(PARAM, doc='Upper limit on control rate before 2150', default=1., sow=0)
        self.limmiu = self.new(PARAM, doc='Upper limit on control rate after 2150', default=1.2, sow=0)
        self.limmiu1 = self.new(PARAM, doc='Upper limit on control rate after 2200', default=1., sow=0)
        self.tnopol = self.new(PARAM, doc='Period before which no emissions controls base', default=45, sow=0)
        self.cprice0 = self.new(PARAM, doc='Initial base carbon price (2010$ per tCO2)', default=2., sow=0)
        self.gcprice = self.new(PARAM, doc='Growth rate of base carbon price per year', default=.02, sow=0)
        ## Scaling and inessential parameters (so that first period's consumption =1 and PV cons = PV utilty)
        self.scale1 = self.new(PARAM, doc='Multiplicative scaling coefficient', default=0.0302455265681763, sow=0)
        self.scale2 = self.new(PARAM, doc='Additive scaling coefficient', default=-10993.704, sow=0)


        #def _init_sets(self):

        # SETS
        ## Param index
        self.p = self.new(RANGESET, 1, self.tnum, doc='Set of parameters for parametric specifications of controls')
        ## Controls
        self.u = self.new(SET, initialize=['MIU', 'S'], doc='Set of controls for parametric specifications', ordered=True)
        ## Targets
        self.x = self.new(SET, initialize=['TATM', 'MAT'], doc='Set of variables on which a target is prescribed', ordered=True)
        ## Target directions
        self.tgtdir = self.new(SET, initialize=[TgtType.LOWER, TgtType.UPPER], doc='Set of target directions', ordered=True)


        # def _init_params(self):

        # PARAMETERS - SCALAR / COMPUTED
        self.optlrsav = self.new(PARAM, doc='Optimal long-run savings rate used for transversality', sow=0,
                              initialize=lambda m: (m.dk + .004)/(m.dk + .004*m.elasmu + m.prstp)*m.gama)
        self.sig0 = self.new(PARAM, doc='Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)',
                          initialize=lambda m: m.e0/(m.q0*(1-m.miu0)))
        self.b11 = self.new(PARAM, doc='Carbon cycle transition matrix',
                         initialize=lambda m: 1 - m.b12)
        self.b21 = self.new(PARAM, doc='Carbon cycle transition matrix',
                         initialize=lambda m: m.b12 * m.mateq / m.mueq)
        self.b22 = self.new(PARAM, doc='Carbon cycle transition matrix',
                         initialize=lambda m: 1 - m.b21 - m.b23)
        self.b32 = self.new(PARAM, doc='Carbon cycle transition matrix',
                         initialize=lambda m: m.b23 * m.mueq / m.mleq)
        self.b33 = self.new(PARAM, doc='Carbon cycle transition matrix',
                         initialize=lambda m: 1 - m.b32)
        self.a20 = self.new(PARAM, doc='Initial damage quadratic term',
                         initialize=lambda m: m.a2)
        self.lam = self.new(PARAM, doc='Climate self parameter',
                         initialize=lambda m: m.fco22x/m.t2xco2, eval_just_once=False)

        # PARAMETERS - META
        self.ctrltype = self.new(PARAM, self.u, doc='Type of parameterization for each control u',
                              default=Ctrl.TYPE_FREE, sow=0)
        self.ctrlparam = self.new(PARAM, self.u, self.p, doc='Value of parameter p for control u parameterization', default=np.nan, sow=0)
        self.tgttype = self.new(PARAM, self.x, self.tgtdir, doc='Type of target', default=TgtType.TYPE_NONE, sow=0)
        self.tgtparam = self.new(PARAM, self.x, self.tgtdir, self.p, doc='Value of parameter p in target constraint for variable x', default=np.nan, sow=0)
        self.damfunc = self.new(PARAM, doc='Damage function', default=Damages.NORDHAUS, sow=0)

        # PARAMETERS - OTHER
        self.l = self.new(PARAM, self.t, doc='Level of population and labor',
                       initialize=lambda m, t: m.l[t - 1] * pow(m.popasym / (m.l[t - 1]), m.popadj) if t > 1 else m.pop0)
        self.ga = self.new(PARAM, self.t, doc='Growth rate of productivity from',
                        initialize=lambda m, t: m.ga0 * m.exp(-m.dela * m.tstep * ((t - 1))))
        self.al = self.new(PARAM, self.t, doc='Level of total factor productivity',
                        initialize=lambda m, t: m.al[t - 1] / (1 - m.ga[t - 1]) if t > 1 else m.a0)
        self.gsig = self.new(PARAM, self.t, doc='Change in sigma (cumulative improvement of energy efficiency)',
                          initialize=lambda m, t: m.gsig[t - 1] * pow(1 + m.dsig, m.tstep) if t > 1 else m.gsigma1)
        self.sigma = self.new(PARAM, self.t, doc='CO2-equivalent-emissions output ratio',
                           initialize=lambda m, t: m.sigma[t - 1] * m.exp(m.gsig[t - 1] * m.tstep) if t > 1 else m.sig0)
        self.pbacktime = self.new(PARAM, self.t, doc='Backstop price',
                               initialize=lambda m, t: m.pback * pow(1 - m.gback, t - 1))
        self.cost1 = self.new(PARAM, self.t, doc='Adjusted cost for backstop',
                           initialize=lambda m, t: m.pbacktime[t] * m.sigma[t] / m.expcost2 / 1e3)
        self.etree = self.new(PARAM, self.t, doc='Emissions from deforestation',
                           initialize=lambda m, t: m.eland0 * pow(1 - m.deland, t - 1))
        self.cumetree = self.new(PARAM, self.t, doc='Cumulative from land',
                              initialize=lambda m, t: m.cumetree[t - 1] + m.etree[t - 1] * m.tstep * m.e2cume if t > 1 else m.cumetree0)
        self.rr = self.new(PARAM, self.t, doc='Average utility social discount rate',
                        initialize=lambda m, t: 1. / pow(1 + m.prstp, m.tstep * (t - 1)))
        self.fex1fac = self.new(PARAM, self.t, doc='Parameter for exogenous forcing',
                             initialize=lambda m, t: (m.year[t] - m.baseyear) / (2100 - m.baseyear) if m.year[t] < 2100  else 1.)
        self.forcoth = self.new(PARAM, self.t, doc='Exogenous forcing for other greenhouse gases',
                             initialize=lambda m, t: m.fex0 + m.fex1fac[t] * (m.fex1 - m.fex0))
        self.cpricebase = self.new(PARAM, self.t, doc='Carbon price in base case',
                                initialize=lambda m, t: m.cprice0 * pow(1 + m.gcprice, m.tstep * (t - 1)))


        #def _init_variables(self):

        # VARIABLES - BOUNDS w/ DEPENDENT INIT
        self.FORC = self.new(VAR, self.t, doc='Increase in radiative forcing (watts per m2 from 1900)',
                        **vbounderwinit((None, None)))
        self.E = self.new(VAR, self.t, doc='Total CO2 emissions (GtCO2 per year)',
                     **vbounderwinit((None, None)))
        self.EIND = self.new(VAR, self.t, doc='Industrial emissions (GtCO2 per year)',
                        **vbounderwinit((None, None)))
        self.EABAT = self.new(VAR, self.t, doc='Abated industrial emissions (GtCO2 per year)',
                        **vbounderwinit((None, None)))
        self.C = self.new(VAR, self.t, doc='Consumption (trillions 2005 US dollars per year)',
                     **vbounderwinit((2., None)))
        self.CPC = self.new(VAR, self.t, doc='Per capita consumption (thousands 2005 USD per year)',
                       **vbounderwinit((1e-2, None)))
        self.I = self.new(VAR, self.t, doc='Investment (trillions 2005 USD per year)',
                     **vbounderwinit((0., None)))
        self.RI = self.new(VAR, self.t, doc='Real interest rate (per annum)',
                      **vbounderwinit((None, None)))
        self.Y = self.new(VAR, self.t, doc='Gross world product net of abatement and damages (trillions 2005 USD per year)',
                     **vbounderwinit((0., None)))
        self.YGROSS = self.new(VAR, self.t, doc='Gross world product GROSS of abatement and damages (trillions 2005 USD per year)',
                          **vbounderwinit((0., None)))
        self.YNET = self.new(VAR, self.t, doc='Output net of damages equation (trillions 2005 USD per year)',
                        **vbounderwinit((None, None)))
        self.DAMAGES = self.new(VAR, self.t, doc='Damages (trillions 2005 USD per year)',
                           **vbounderwinit((None, None)))
        self.DAMFRAC = self.new(VAR, self.t, doc='Damages as fraction of gross output',
                           **vbounderwinit((None, None)))
        self.ABATECOST = self.new(VAR, self.t, doc='Cost of emissions reductions  (trillions 2005 USD per year)',
                             **vbounderwinit((None, None)))
        self.MCABATE = self.new(VAR, self.t, doc='Marginal cost of abatement (2005$ per ton CO2)',
                           **vbounderwinit((None, None)))
        self.CCATOT = self.new(VAR, self.t, doc='Total carbon emissions (GtC)',
                          **vbounderwinit((None, None)))
        self.PERIODU = self.new(VAR, self.t, doc='One period utility function',
                           **vbounderwinit((None, None)))
        self.CPRICE = self.new(VAR, self.t, doc='Carbon price (2005$ per ton of CO2)',
                          **vbounderwinit((None, None)))
        self.CEMUTOTPER = self.new(VAR, self.t, doc='Period utility',
                              **vbounderwinit((None, None)))
        self.MIU = self.new(VAR, self.t, doc='Emission control rate GHGs',
                       **vbounderwinit(miubounder, tfirstfixer('MIU', 'miu0')))

        # VARIABLES - BOUNDS w/ INDEPENDENT INIT
        self.CCA = self.new(VAR, self.t, doc='Cumulative industrial carbon emissions (GTC)',
                       bounds=lambda m, t: (0., m.fosslim),
                       initialize=tfirstfixer('CCA', 'cca0'))
        self.MAT = self.new(VAR, self.t, doc='Carbon concentration increase in atmosphere (GtC from 1750)',
                       bounds=(280*2.13, None), # pre-industrial level are 280 ppm
                       initialize=tfirstfixer('MAT', 'mat0'))
        self.MU = self.new(VAR, self.t, doc='Carbon concentration increase in shallow oceans (GtC from 1750)',
                      bounds=(1e2, None),
                      initialize=tfirstfixer('MU', 'mu0'))
        self.ML = self.new(VAR, self.t, doc='Carbon concentration increase in lower oceans (GtC from 1750)',
                      bounds=(1e3, None),
                      initialize=tfirstfixer('ML', 'ml0'))
        self.TATM = self.new(VAR, self.t, doc='Increase temperature of atmosphere (degrees C from 1900)',
                        bounds=(1e-2, 12.),
                        initialize=tfirstfixer('TATM', 'tatm0'))
        self.TOCEAN = self.new(VAR, self.t, doc='Increase temperature of lower oceans (degrees C from 1900)',
                          bounds=(-1., 20.),
                          initialize=tfirstfixer('TOCEAN', 'tocean0'))
        self.K = self.new(VAR, self.t, doc='Capital stock (trillions 2005 US dollars)',
                     bounds=(1., None),
                     initialize=tfirstfixer('K', 'k0'))
        self.S = self.new(VAR, self.t, doc='Gross savings rate as fraction of gross world product',
                     bounds=(0., None), sow=0)
                     #initialize=tfixer('S', 'optlrsav', lambda m,t: m.year[t]>2150))

        #def _init_equations(self):

    @staticmethod
    def npv(m, v):
        tgood = [t for t in m.t if m.year[t] >= m.baseyear]
        vv = getattr(m, v)
        return sum(vv[t] * 1 / pow(1.05, m.year[t] - m.baseyear) for t in tgood)

    @staticmethod
    def npv_gdp(m, v):
        return 100*(Dice.npv(m, v))/(Dice.npv(m, 'YGROSS'))

    @staticmethod
    def cbge_v1(m, w):
        wref = sum(((pow(m.YGROSS[t] * (1 - m.S[t]) * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1) * m.l[t] * m.rr[t] for t in m.t[1:])
        return 100.*(pow(w/wref, 1/(1 - m.elasmu)) - 1)

    @staticmethod
    def welfare(m, cc):
        return sum(pow(cc[t]*1e3/m.l[t], 1-m.elasmu)/(1-m.elasmu)*m.l[t]*m.rr[t] for t in m.t[1:])

    @staticmethod
    def cbge(m, w):
        wref = Dice.welfare(m, m.YGROSS * ((1 - m.S)[:,np.newaxis]))
        return -100.*(pow(np.mean(w)/np.mean(wref), 1/(1 - m.elasmu)) - 1)

    @staticmethod
    def welfare_ref(m):
        return Dice.welfare(m, m.YGROSS * ((1 - m.S)[:, np.newaxis]))

    @staticmethod
    def cbge_quantile(m, w, q=0.95):
        wref = Dice.welfare_ref(m)
        return np.quantile(-100.*(pow(w/wref, 1/(1 - m.elasmu)) - 1), q)

    @staticmethod
    def cbge_mitcost_v1(m):
        w = sum(((pow((m.YGROSS[t] - m.ABATECOST[t]) * (1 - m.S[t]) * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1) * m.l[t] * m.rr[t] for t in m.t[1:])
        return Dice.cbge_v1(m, w)

    @staticmethod
    def cbge_damcost_v1(m):
        w = sum(((pow(m.YGROSS[t]*(1 - m.DAMFRAC[t]) * (1 - m.S[t]) * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1) * m.l[t] * m.rr[t] for t in m.t[1:])
        return Dice.cbge_v1(m, w)

    @staticmethod
    def cbge_mitcost(m):
        w = Dice.welfare(m, (m.YGROSS - m.ABATECOST) * ((1 - m.S)[:,np.newaxis]))
        return Dice.cbge(m, w)

    @staticmethod
    def cbge_damcost(m):
        w = Dice.welfare(m, m.YGROSS*(1 - m.DAMFRAC) * ((1 - m.S)[:,np.newaxis]))
        return Dice.cbge(m, w)
    
    @staticmethod
    def welfare_damcost(m):
        return Dice.welfare(m, m.YGROSS*(1 - m.DAMFRAC) * ((1 - m.S)[:,np.newaxis]))

    @staticmethod
    def bge_damcost_q95(m):
        w = Dice.welfare(m, m.YGROSS*(1 - m.DAMFRAC) * ((1 - m.S)[:,np.newaxis]))
        return Dice.cbge_quantile(m, w, q=0.95)

    def _body_eqs(self):
        # EQUATIONS - TIME
        ## Controls equations
        self.controlmiueq = self.new(CONSTRAINT, self.MIU, Ctrl.controller('MIU'), sow=0)
        self.controlseq = self.new(CONSTRAINT, self.S, Ctrl.controller('S'), sow=0)
        self.maxtemp = self.new(CONSTRAINT, self.TATM, TgtType.limiter('TATM', TgtType.UPPER), op=operator.le)
        self.mintemp = self.new(CONSTRAINT, self.TATM, TgtType.limiter('TATM', TgtType.LOWER), op=operator.ge)
        self.maxconc = self.new(CONSTRAINT, self.MAT, TgtType.limiter('MAT', TgtType.UPPER), op=operator.le)
        self.minconc = self.new(CONSTRAINT, self.MAT, TgtType.limiter('MAT', TgtType.LOWER), op=operator.ge)
        ## Climate and carbon cycle
        self.mmat = self.new(EQUATION, self.MAT, lambda m, t: m.MAT[t - 1] * m.b11 + m.MU[t - 1] * m.b21 + (m.E[t - 1] * m.tstep * m.e2cume) if t > 1 else None)
        self.mml = self.new(EQUATION, self.ML, lambda m, t: m.ML[t - 1] * m.b33 + m.MU[t - 1] * m.b23 if t > 1 else None)
        self.mmu = self.new(EQUATION, self.MU, lambda m, t: m.MAT[t - 1] * m.b12 + m.MU[t - 1] * m.b22 + m.ML[t - 1] * m.b32 if t > 1 else None)
        self.force = self.new(EQUATION, self.FORC, lambda m, t: m.fco22x * (m.log((m.MAT[t] / 588.000)) / m.log(2)) + m.forcoth[t])
        self.tatmeq = self.new(EQUATION, self.TATM, lambda m, t: (m.TATM[t - 1] + m.c1 * ((m.FORC[t] - (m.fco22x / m.t2xco2) * m.TATM[t - 1]) - (m.c3 * (m.TATM[t - 1] - m.TOCEAN[t - 1])))) if t > 1 else None)
        self.toceaneq = self.new(EQUATION, self.TOCEAN, lambda m, t: m.TOCEAN[t - 1] + m.c4 * (m.TATM[t - 1] - m.TOCEAN[t - 1]) if t > 1 else None)
        ## Economic variables
        self.kk = self.new(EQUATION, self.K, lambda m, t: pow(1 - m.dk, m.tstep) * m.K[t - 1] + m.tstep * m.I[t - 1] if t > 1 else None)
        self.damfraceq = self.new(EQUATION, self.DAMFRAC, lambda m, t: Damages.damfrac(m,t))
        self.ygrosseq = self.new(EQUATION, self.YGROSS, lambda m, t: (m.al[t] * pow(m.l[t] / 1e3, 1. - m.gama)) * pow(m.K[t], m.gama))
        self.dameq = self.new(EQUATION, self.DAMAGES, lambda m, t: m.YGROSS[t] * m.DAMFRAC[t])
        self.yneteq = self.new(EQUATION, self.YNET, lambda m, t: m.YGROSS[t] * (1 - m.DAMFRAC[t]))
        self.abateeq = self.new(EQUATION, self.ABATECOST, lambda m, t: m.YGROSS[t] * m.cost1[t] * pow(m.MIU[t], m.expcost2))
        self.mcabateeq = self.new(EQUATION, self.MCABATE, lambda m, t: m.pbacktime[t] * pow(m.MIU[t], m.expcost2 - 1))
        self.carbpriceeq = self.new(EQUATION, self.CPRICE, lambda m, t: m.pbacktime[t] * pow(m.MIU[t], m.expcost2 - 1))
        self.yy = self.new(EQUATION, self.Y, lambda m, t: m.YNET[t] - m.ABATECOST[t])
        self.seq = self.new(EQUATION, self.I, lambda m, t: m.S[t] * m.Y[t])
        self.cc = self.new(EQUATION, self.C, lambda m, t: m.Y[t] - m.I[t])
        self.cpce = self.new(EQUATION, self.CPC, lambda m, t: 1e3 * m.C[t] / m.l[t])

        #self.rieq = self.new(EQUATION, self.RI, lambda m, t: (1 + m.prstp) * pow(m.CPC[t+1] / m.CPC[t], m.elasmu / m.tstep) - 1 if t < m.tnum else None)

        ## Emissions
        self.eindeq = self.new(EQUATION, self.EIND, lambda m, t: m.sigma[t] * m.YGROSS[t] * (1 - (m.MIU[t])))
        self.eeq = self.new(EQUATION, self.E, lambda m, t: m.EIND[t] + m.etree[t])
        self.eabateq = self.new(EQUATION, self.EABAT, lambda m, t: m.sigma[t] * m.YGROSS[t] * m.MIU[t])
        self.ccacca = self.new(EQUATION, self.CCA, lambda m, t: m.CCA[t - 1] + m.EIND[t - 1] * m.tstep * m.e2cume if t > 1 else None)
        self.ccatoteq = self.new(EQUATION, self.CCATOT, lambda m, t: m.CCA[t] + m.cumetree[t])

        ## Utility related
        self.periodueq = self.new(EQUATION, self.PERIODU, lambda m, t: (pow(m.C[t] * 1e3 / m.l[t], 1 - m.elasmu) - 1) / (1 - m.elasmu) - 1)
        self.cemutotpereq = self.new(EQUATION, self.CEMUTOTPER, lambda m, t: m.PERIODU[t] * m.l[t] * m.rr[t])

        # OBJECTIVES
        # Main
        self.MAX_UTIL = self.new(OBJECTIVE, doc='Maximize expected utility',
                                 sense=maximize, sow_reduce=np.mean,
                              rule=lambda m : m.tstep * m.scale1 * m.summation(m.CEMUTOTPER) + m.scale2)
        # Others (deactivated in single-obj optimization)
        self.MAX_UTIL_BGE = self.new(OBJECTIVE, doc='Maximize balanced growth equivalent utility', only=MODE_SIM,
                                 sense=maximize, sow_reduce=np.mean,
                              rule=lambda m : Dice.cbge(m, Dice.welfare(m, m.C)))
        self.MIN_LOSS_UTIL_BGE = self.new(OBJECTIVE, doc='Minimize balanced growth equivalent utility loss', only=MODE_SIM,
                                 sense=minimize, sow_reduce=np.mean,
                              rule=lambda m : Dice.cbge(m, Dice.welfare(m, m.C)))
        self.MIN_LOSS_UTIL_95Q_BGE = self.new(OBJECTIVE, doc='Minimize balanced growth equivalent utility loss', only=MODE_SIM,
                                 sense=minimize,
                              rule=lambda m : Dice.cbge_quantile(m, Dice.welfare(m, m.C), q=0.95))
        self.MIN_LOSS_UTIL_90Q_BGE = self.new(OBJECTIVE, doc='Minimize balanced growth equivalent utility loss', only=MODE_SIM,
                                 sense=minimize,
                              rule=lambda m : Dice.cbge_quantile(m, Dice.welfare(m, m.C), q=0.90))
        self.MIN_TATM2100 = self.new(OBJECTIVE, doc='Minimize temperature increase in 2100',
                                     sense=minimize, sow_reduce=np.mean, only=MODE_SIM,
                              rule=lambda m : sum(m.TATM[t] for t in m.t if m.year[t]==2100))
        self.MIN_MIU2020 = self.new(OBJECTIVE, doc='Minimize abatement in 2020',
                                     sense=minimize, sow_reduce=np.mean, only=MODE_SIM,
                              rule=lambda m : sum((m.MIU[t]-m.MIU[1])/(m.TATM[t]-m.TATM[1]) for t in m.t if m.year[t]==2020))
        self.MIN_MIU2030 = self.new(OBJECTIVE, doc='Minimize abatement in 2030',
                                     sense=minimize, sow_reduce=np.mean, only=MODE_SIM,
                              rule=lambda m : sum((m.MIU[t]-m.MIU[1])/(m.TATM[t]-m.TATM[1]) for t in m.t if m.year[t]==2030))
        self.MIN_MIU2050 = self.new(OBJECTIVE, doc='Minimize abatement in 2050',
                                     sense=minimize, sow_reduce=np.mean, only=MODE_SIM,
                              rule=lambda m : sum((m.MIU[t]-m.MIU[1])/(m.TATM[t]-m.TATM[1]) for t in m.t if m.year[t]==2050))
        self.MAX_REL2C = self.new(OBJECTIVE, doc='Maximize # of SOWs w/ temperature below 2C',
                                  sense=maximize, only=MODE_SIM,
                                  sow_reduce=lambda x: 100*np.count_nonzero(x<=2)/len(x),
                              rule=lambda m : np.max(list(m.TATM[t] for t in m.t[1:]), axis=0))
        self.MIN_MEAN2DEGYEARS = self.new(OBJECTIVE, doc='Minimize 2C years',
                                      sense=minimize, only=MODE_SIM,
                                      sow_reduce=np.mean,
                              rule=lambda m : np.sum((np.maximum(0, m.TATM[1:-10]-2))*m.tstep, axis=0))
        self.MIN_MAX2DEGYEARS = self.new(OBJECTIVE, doc='Minimize 2C years',
                                      sense=minimize, only=MODE_SIM,
                                      sow_reduce=np.max,
                              rule=lambda m : np.sum((np.maximum(0, m.TATM[1:-10]-2))*m.tstep, axis=0))
        self.MIN_NPVMITCOST = self.new(OBJECTIVE, doc='Minimize NPV mitigation costs',
                                       sense=minimize, only=MODE_SIM,
                                       sow_reduce=np.mean,
                              rule=lambda m : Dice.npv_gdp(m, 'ABATECOST'))
        self.MIN_CBGEMITCOSTv1 = self.new(OBJECTIVE, doc='Minimize CBGE mitigation costs',
                                       sense=minimize, only=MODE_SIM, sow_reduce=np.mean,
                              rule=Dice.cbge_mitcost_v1)
        self.MIN_CBGEMITCOST = self.new(OBJECTIVE, doc='Minimize CBGE mitigation costs',
                                       sense=minimize, only=MODE_SIM, #sow_reduce=None,
                              rule=Dice.cbge_mitcost)
        self.MIN_NPVDAMCOST = self.new(OBJECTIVE, doc='Minimize NPV damage costs',
                                       sense=minimize, only=MODE_SIM,
                                       sow_reduce=np.mean,
                              rule=lambda m : Dice.npv_gdp(m, 'DAMAGES'))
        self.MIN_CBGEDAMCOSTv1 = self.new(OBJECTIVE, doc='Minimize CBGE damage costs',
                                       sense=minimize, only=MODE_SIM, sow_reduce=np.mean,
                              rule=Dice.cbge_damcost_v1)
        self.MIN_CBGEDAMCOST = self.new(OBJECTIVE, doc='Minimize CBGE damage costs',
                                       sense=minimize, only=MODE_SIM, #sow_reduce=None,
                              rule=Dice.cbge_damcost)
        self.MIN_PEAKABATRATE = self.new(OBJECTIVE, doc='Minimize peak in abatement upscaling', only=MODE_SIM,
                                 sense=minimize, sow_reduce=np.max,
                              rule=lambda m : 100.*np.max(np.abs(m.MIU[1:]-shift(m.MIU[1:],1))[1:],axis=0)/(m.tstep))
        self.MIN_Q95MAXTEMP = self.new(OBJECTIVE, doc='Minimize 95th percentile max temperature', only=MODE_SIM,
                                 sense=minimize, sow_reduce=partial(np.percentile, q=95),
                              rule=lambda m : np.max(m.TATM[1:-10],axis=0))
        self.MIN_Q95DAMCOST = self.new(OBJECTIVE, doc='Minimize CBGE damage costs',
                                       sense=minimize, only=MODE_SIM, #sow_reduce=partial(np.percentile, q=95),
                              rule=Dice.bge_damcost_q95)


        # BRUSHES
        self.PEAKABATRATE_4PCTYR_MAX = self.new(BRUSH, doc='Minimize peak in abatement upscaling',
                                                only=MODE_SIM, sow_reduce=np.max,
                              rule=lambda m: (100.*np.max(np.abs(m.MIU[1:]-shift(m.MIU[1:],1))[1:],axis=0)/(m.tstep)-4.01))
        self.PEAKABATRATE_4PCTYR_95Q = self.new(BRUSH,
                                                doc='Minimize peak in abatement upscaling', only=MODE_SIM,
                                                sow_reduce=partial(np.percentile, q=95),
                              rule=lambda m: (100.*np.max(np.abs(m.MIU[1:]-shift(m.MIU[1:],1))[1:],axis=0)/(m.tstep)-4.01))


    def _after_solve(self, m):
        years = m['year']
        sum_over_century = lambda x: m['MIU'].rename(years).reindex(range(years.iloc[0], years.iloc[-1]+1)).loc[:2100].interpolate().sum()
        m['PPMATM'] = m['MAT'] / 2.13
        m['EBASE'] = m['sigma'] * m['YGROSS']
        m['TOT_EABAT'] = sum_over_century(m['EABAT'])
        vcosts = [v for v in m.keys() if v[-4:] == 'COST']
        for v in vcosts:
            m['{v}_PCTGDP'.format(v=v)] = m[v].div(m['YGROSS'])
        t = m['year']
        m['NPV_discount'] = (1. / (1. + m['RI'])).pow(m['year']-m['year'].iloc[0])
        m['NPV_C'] = m['C'].mul(m['NPV_discount']).sum()
        for scen in ['bau', 'opt']:
            try:
                yref = Data.load(scen)
                m['NPV_C_DIFF_{scen}'.format(scen=scen.upper())] = yref['NPV_C'] - m['NPV_C']
                m['NPV_C_DIFF_{scen}'.format(scen=scen.upper())] = yref['NPV_C'] - m['NPV_C']
                m['NPV_C_DIFFPCT_{scen}'.format(scen=scen.upper())] = (yref['NPV_C'] - m['NPV_C']) / (yref['NPV_C'])
                m['C_DIFFPCT_{scen}'.format(scen=scen.upper())] = 100.*(yref['C'] - m['C']).div(yref['C'])
                m['CEMUTOTPER_DIFFPCT_{scen}'.format(scen=scen.upper())] = 100. * (yref['CEMUTOTPER'] - m['CEMUTOTPER']).div(yref['CEMUTOTPER'])
            except:
                m['NPV_C_DIFF_{scen}'.format(scen=scen.upper())] = 0 * m['C']
                m['NPV_C_DIFFPCT_{scen}'.format(scen=scen.upper())] = 0 * m['C']


    def calc_bau(self):
        self.set(a2=0)
        dice_opt = Dice(time=Time(start=2015, periods=100, tstep=5),
                        name=self._name, mode=MODE_OPT, setup=self.setup,
                        calib=self._calib, default_sow=0,
                        **self._kwargs)
        self.bau = dice_opt.set_bau().solve()


    def add_learning(self):
        self.mcabate0 = Param(initialize=lambda m: m.pback/m.expcost2) #*(m.miu0**(m.expcost2-1)))
        self.del_component(self.MCABATE)
        self.MCABATE = Var(self.t, bounds=lambda m,t: (None, None),
                           initialize= tfirstfixer('MCABATE', 'mcabate0'))
        self.cumabat0 = Param(doc='Initial cumulative abatament of industrial CO2 (GtC)', default=10)
        self.abatpr = Param(doc='Progress ratio of technologies for CO2 abatement', default=0.85)
        self.CUMABAT = Var(self.t, **vbounderwinit((0., None), tfirstfixer('CUMABAT', 'cumabat0')))
        self.cumabateq = Constraint(self.t, rule=
            lambda m,t: m.CUMABAT[t] == m.CUMABAT[t-1]+m.e2cume*m.tstep*m.EABAT[t] if t>1 else None)
        self.del_component(self.mcabateeq)
        self.mcabateeq = Constraint(self.t, doc='Learning by doing for abat tech', rule=
            lambda m,t: m.MCABATE[t] == m.mcabate0*((m.CUMABAT[t-1]/m.CUMABAT[1])**(-m.abatpr)) if t>1 else None)
        self.del_component(self.abateeq)
        self.abateeq = Constraint(self.t, rule=lambda m, t: m.ABATECOST[t] == 1e-3*m.EABAT[t]*m.MCABATE[t])
        return self


    def set_control_exp(self, v, rate=0.05):
        self.set('ctrltype', {v: Ctrl.TYPE_EXPONENTIAL})
        self.set('ctrlparam', {(v, Ctrl.PAR_RATE): rate})
        return self


    def set_control_logistic(self, v, syear=2020, sfrac=0.1, maxlev=1., rate=0.05):
        self.set('ctrltype', {v: Ctrl.TYPE_LOGISTIC})
        self.set('ctrlparam', {(v,Ctrl.PAR_STARTYEAR): syear,
                               (v,Ctrl.PAR_STARTFRAC): sfrac,
                               (v,Ctrl.PAR_MAXLEVEL): maxlev,
                               (v,Ctrl.PAR_RATE): rate})
        return self


    def set_control_const(self, v):
        self.set('ctrltype', {v: Ctrl.TYPE_CONST})
        return self




    def add_logistic_control(self, vname): #start=2020, speed=0.05, max=1.):
        #setattr(self, f'{vname.lower()}_s±tart', start)
        #setattr(self, f'{vname.lower()}_speed', speed)
        #setattr(self, f'{vname.lower()}_max', max)
        self.add_rule(vname, logisticfixer(vname))
        return self

    def add_exponential_control(self, vname):
        self.add_rule(vname, exponentialfixer(vname))

    def set_scalar(self, p, pval):
        self.set(p, pval)
        return self

    """
    def fix(self, varname, value, force=False, viaparam=True):
        if isinstance(value, str):
            value = Data.load(value)
        if isinstance(value, DiceSolution):
            value = getattr(value.insts[0], varname).values()
            ctype = Ctrl.TYPE_FIX_SERIES
            getval = lambda x: x.value
        elif isinstance(value, Data):
            value = getattr(value, varname)
            ctype = Ctrl.TYPE_FIX_SERIES
            getval = lambda x: x
        elif isinstance(value, int) or isinstance(value, float):
            value = pd.Series([float(value)], index=[1])
            ctype = Ctrl.TYPE_FIX_SCALAR
            getval = lambda x: x
        else:
            raise Exception(f'Value "{value}" not supported')
        if viaparam:
            self.set(self.ctrltype, {varname: ctype})
            self.set(self.ctrlparam, {(varname, i): getval(v) for i,v in value.iteritems()})
        else:
            #value = y.iloc[:,0] if isinstance(y, pd.DataFrame) else y.iloc[0]
            #if isinstance(value, pd.DataFrame):
            #    value = value.iloc[:,0]
            #assert isinstance(value, pd.Series)
            super().fix(varname, value, force=force)
        return self
    """

    def add_stabilization(self, what='TATM', after=2100):
        self.stabeq = Constraint(self.t, rule=lambda m,t: getattr(m, what)[t] <= getattr(m, what)[t-1] if m.year[t] > after else None)
        return self


    def set_target(self, what, howmuch, bywhen, how=TgtType.UPPER):
        self.set(self.tgttype, {(what, how): TgtType.TYPE_ACTIVE})
        self.set(self.tgtparam, {(what, how, TgtType.PAR_STARTYEAR): bywhen,
                                 (what, how, TgtType.PAR_THRES): howmuch})
        return self


    def set_policy(self, pol):
        if pol == 'bau':
            self.set(a2=0.)
        elif pol == '2deg':
            self.set_target('TATM', 2, 2100)
        elif pol == 'resto2100':
            self.set_target('MAT', 280 * 2.13, 2100, TgtType.UPPER)
        elif pol == 'resto2100and2deg':
            self.set_target('TATM', 2, 2100)
            self.set_target('MAT', 280 * 2.13, 2100, TgtType.UPPER)
            #self.set_target('MAT', 280 * 2.13, 2100, TgtType.LOWER)
        elif pol == 'resto2050':
            self.set_target('MAT', 280 * 2.13, 2050, TgtType.UPPER)
        elif pol == 'resto2050and2deg':
            self.set_target('TATM', 2, 2050)
            self.set_target('MAT', 280 * 2.13, 2050, TgtType.UPPER)
            #self.set_target('MAT', 280 * 2.13, 2050, TgtType.LOWER)
        elif pol == 'opt':
            pass
        else:
            raise Exception('Policy "{pol}" not recognized'.format(pol=pol))
        return self

    def set_bau(self):
        return self.set_policy('bau')


class Dice(DiceBase):

    def __init__(self, mode, endyear=2200, **kwargs):
        if not 'setup' in kwargs:
            kwargs['setup'] = defaultdict(dict)
        kwargs['setup']['yearccs'] = kwargs['setup'].get('yearccs', {None: 2020})
        super().__init__(mode=mode, endyear=endyear, **kwargs)



class DiceMulti(Dice):
    def __init__(self, *args, time=None, **kwargs):
        if time is None:
            time = Time(start=2015, end=2100, tstep=5)
        assert 'mode' not in kwargs
        kwargs['mode'] = MODE_SIM
        assert 'vin' not in kwargs
        kwargs['vin'] = ['MIU']
        assert 'vout' not in kwargs
        kwargs['vout'] = ['NPVMITCOST', 'TATM2100']
        super().__init__(*args, time=time, **kwargs)

    """
    def __init__(self, name=None, bau=None, **kwargs):
        self.set('endyear', 2105)
        super().__init__(name=name)
        # Fix savings rate to bau
        if bau is None:
            bau = Dice('BAU').apply(scenBau).solve(tee=False)
        self.bau = bau
        self.add_fix('S', bau, force=True)
        
    def solve(self):
        logger.info('Solving in SINGLE-OBJECTIVE optimization mode')
        return self.duplicate(setup={'endyear':2200}).solve()
    """


