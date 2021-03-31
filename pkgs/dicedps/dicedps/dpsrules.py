from paradigm import Model, Time, MODE_SIM, RANGESET, VAR, EQUATION, PARAM
from paradoeclim import get_hist_temp_data
import re
import numpy as np


class MiuController(Model):

    def __init__(self, adapter, **kwargs):
        dice = adapter._dice
        time = Time(dice.time.start, dice.time.end, tstep=dice.time.tstep)
        self._dice = dice
        self._adapter = adapter
        super().__init__(time=time, mode=MODE_SIM, default_sow=dice._default_sow, **kwargs)


class MiuProportionalController(MiuController):

    def __init__(self, dice, time=None, **kwargs):
        if time is None:
            time = Time(dice.time.start, dice.time.end, tstep=dice.time.tstep)
        self._dice = dice
        super().__init__(time, mode=MODE_SIM, vin=['X'], **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')

    def _body(self):
        self.u = self.new(RANGESET, 1, 1)
        self.X = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[0,1], sow=0)
        self.TATM = self._dice.TATM  # self.new(VAR, self._dice.t, doc='TATM <- DICE')
        self.MIU = self._dice.MIU  # self.new(VAR, self._dice.t, doc='MIU -> DICE')
        self.MIU_bounds = self._dice.MIU_bounds  # self.new(VAR, self._dice.t, doc='MIU -> DICE')

    @staticmethod
    def eqmiu_rule(m, t):
        miu = np.minimum(m.MIU_bounds[t,1], m.X[1]*m.TATM[max(1,t-1)])
        return miu

    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, MiuProportionalController.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)


def get_temp_from_doeclim(m, t):
    t = m._adapter._tdice2doeclim[t]
    # Average over 5 years
    tprev = max(1, t-5)
    #tdelta = tprev - max(1,tprev-2)
    #return np.mean(m._adapter._doeclim.temp[(tprev-tdelta):(tprev+tdelta+1)], axis=0)
    return m._adapter._doeclim.temp[tprev]


def get_temp_diff_from_doeclim(m, t):
    return get_temp_from_doeclim(m, t) - get_temp_from_doeclim(m, max(1,t-1))


signal2func = {
    'X': get_temp_from_doeclim,
    'dX': get_temp_diff_from_doeclim,
    'T': (lambda m, t: (m.TATM[max(1,t-1)])),
    'dT': (lambda m, t: (m.TATM[max(1,t-1)]-m.TATM[max(1,t-2)])),
    'G': (lambda m, t: (m._dice.YGROSS[max(1,t-1)])),
    'B': (lambda m, t: ((m.year[t]>=m._dice.yearccs) & (m.year[t]<m._dice.yearnoccs)).astype(float))
}


signal2bounds = {
    'B': [0, 1],
    'G': [0, 5e3],
    'T': [-7, 20],
    'dT': [-1, 1],
}
signal2bounds['X'] = signal2bounds['T']
signal2bounds['dX'] = signal2bounds['dT']


def dpslab2signals(s):
    signals = []
    while len(s)>0:
        bfound = False
        for i in [2,1]:
            if s[:i] in signal2func:
                signals.append(s[:i])
                s = s[i:]
                bfound = True
                break
        assert bfound, f'Unknown signal in {s}'
    return signals



def miu2dargs(miu):
    """Convert miu label into args dictionary."""
    if miu.startswith('time'):
        return {'miu': miu}
    try:
        assert int(miu[-3:])
        return {'miu': miu[3:-3], 'rbfn': miu[-3], 'maxrate': miu[-2], 'thres': miu[-1]}
    except:
        try:
            assert int(miu[-2:])
            return {'miu': miu[3:-2], 'rbfn': miu[-2], 'thres': miu[-1]}
        except:
            assert int(miu[-1])
            return {'miu': miu[3:-1], 'rbfn': miu[-1]}


def miulab2nvars(miulab):

    if miulab.startswith('time'):
        return int((2250-2015)/5 + 1 - 1)
    miudict = miu2dargs(miulab)
    nsig = len(dpslab2signals(miudict['miu']))
    nrbf = int(miudict['rbfn'])
    nthres = int(miudict['thres'])
    if nthres > 1:
        nthres = 0
    return nrbf+2*nsig*nrbf+2*nthres


def dpslab2nvars(s):
    if s.startswith('time'):
        return 47
    if s[:3] == 'rbf':
        try:
            snum = re.search(r'(\d\d+)$', s).group(0)
        except:
            snum = s[-1]
        nsig = len(dpslab2signals(s[3:-len(snum)]))
        nrbf = int(snum[-1])
        return nrbf+2*nsig*nrbf
    raise Exception(f'{s} not recognized')


class MiuRBFController(MiuController):

    MUM_LIN, MUM_FIX = range(2)

    def __init__(self,
                 adapter,
                 n=5,
                 thres=0,
                 signals=['T', 'dT'],
                 max_rate=4,
                 miu_update_tstep=1,
                 miu_update_method=MUM_FIX,
                 **kwargs):
        self._nrbf = n
        self._signals = signals
        self._nsignals = len(signals)
        self._get_signals = lambda m, t: [(signal2func[s](self, t)) for s in signals]
        self._default_max_rate = max_rate
        self._miu_update_tstep = miu_update_tstep
        self._miu_update_method = miu_update_method
        self._thres = thres
        vin = ['W'] + [f'B{s}' for s in signals] + [f'R{s}' for s in signals]
        if thres:
            vin += ['TEMP_THRES','ABAT_RATE_AFTER_THRES']
        super().__init__(adapter,
                         vin=vin, **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')

    def _body(self):
        self.u = self.new(RANGESET, 1, self._nrbf)
        self.max_rate = self.new(PARAM, doc='Maximum annual increase in MIU [% points]',
                                 default=self._default_max_rate, sow=0)
        self.miu_update_tstep = self.new(PARAM, doc='Time steps between MIU updates [# time steps]',
                                         default=self._miu_update_tstep, sow=0)
        self.miu_update_method = self.new(PARAM, doc='Interpolation method between MIU updates [# time steps]',
                                         default=self._miu_update_method, sow=0)
        self.W = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[1e-5,1.], sow=0)
        self.Bs = []
        self.Rs = []
        self.signals_bounds = []
        for s in self._signals:
            currB = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[0.01, 1], sow=0)
            setattr(self, f'B{s}', currB)
            self.Bs.append(currB)
            currR = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[-1, 1], sow=0)
            setattr(self, f'R{s}', currR)
            self.Rs.append(currR)
            self.signals_bounds.append(signal2bounds[s])
            setattr(self, s, self.new(VAR, self.t, doc=f'Signal {s}', bounds=signal2bounds[s]))
            #for v in signal2vars[s]:
            #    setattr(self, v, getattr(self._dice, v))
        self.MIU = self._dice.MIU
        self.TATM = self._dice.TATM
        self.MIU_bounds = self._dice.MIU_bounds
        if self._thres>0:
            self.TEMP_THRES = self.new(VAR, doc='MIU[t] = X*TATM[T]', bounds=[1e-5,20.], sow=0)
            self.ABAT_RATE_AFTER_THRES = self.new(VAR, doc='MIU[t] = X*TATM[T]', bounds=[0.,self._default_max_rate], sow=0)
        else:
            self.TEMP_THRES = self.new(PARAM, doc='MIU[t] = X*TATM[T]', default=100, sow=0)
            self.ABAT_RATE_AFTER_THRES = self.new(PARAM, doc='MIU[t] = X*TATM[T]', default=self._default_max_rate, sow=0)

    @staticmethod
    def rbf(m, t, xlist):
        if (max(0, t - 2) % m.miu_update_tstep) != 0:
            if m.miu_update_method == MiuRBFController.MUM_FIX:
                miu01 = m.MIU[t-1]
            elif m.miu_update_method == MiuRBFController.MUM_LIN:
                miu01 = m.MIU[t-1]+(m.MIU[t-1]-m.MIU[max(t-2,1)])
            else:
                assert False, f'miu_update_method = {m.miu_update_method} not supported'
        else:
            n = len(m.W)-1
            bthres = m.TATM[t-1]>m.TEMP_THRES
            miu01rbf = sum([m.W[j]*np.exp(
                        -sum([((((x-xbnds[0])/(xbnds[1]-xbnds[0]))-R[j])/B[j])**2 for x, xbnds, R, B in zip(xlist, m.signals_bounds, m.Rs, m.Bs)]))
                         for j in range(1,n+1)])
            miu01 = bthres*(m.MIU[t-1]+m.ABAT_RATE_AFTER_THRES/100.*m.tstep) + (1-bthres)*1.2*miu01rbf
        # / sum([m.W[j] for j in range(1,n+1)]))
        #miu = (m.MIU_bounds[t,1]-m.MIU_bounds[t,0])*miu01+m.MIU_bounds[t,0]
        miu = np.maximum(
            np.minimum(
                m.MIU_bounds[t,1],
                    miu01),
            m.MIU_bounds[t,0])
        if m.max_rate > 0:
            assert t>1
            miu = np.maximum(
                np.minimum(
                    m.MIU[t-1]+m.max_rate/100.*m.tstep,
                    miu),
                np.minimum(1., m.MIU[t-1]))
        return miu

    def eqmiu_rule(self, m, t):
        xlist = self._get_signals(m, t)
        miu = MiuRBFController.rbf(m, t, xlist)
        return miu

    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, self.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)

    def plot(self):
        import matplotlib.pylab as plt
        if len(self.signals_bounds)==1:
            fig, ax = plt.subplots(1,1,figsize=(6,4))
            xs = np.linspace(*(self.signals_bounds[0]))
            mius0 = np.zeros_like(xs)
            mius1 = np.zeros_like(xs)
            for i in range(len(xs)):
                mius0[i] = MiuRBFController.rbf(self, 1, [xs[i]])
                mius1[i] = MiuRBFController.rbf(self, len(self.time._range), [xs[i]])
            ax.plot(xs, mius0, label=f't={self.time._range[0]}')
            ax.plot(xs, mius1, label=f't={self.time._range[-1]}')
            ax.legend()


class MiuRbfTController(MiuRBFController):

    def __init__(self, dice, n=5, **kwargs):
        super().__init__(dice, n=n, signals=['T'], **kwargs)


class MiuRbfTdTController(MiuRBFController):

    def __init__(self, dice, n=5, **kwargs):
        super().__init__(dice, n=n, signals=['T', 'dT'], **kwargs)


class MiuRbfTdTGBController(MiuRBFController):

    def __init__(self, dice, n=5, **kwargs):
        super().__init__(dice, n=n, signals=['T', 'dT', 'G', 'B'], **kwargs)


def args2dpsclass(args):
    miu2class = {'dps1': MiuProportionalController,
                 'dps5': MiuPolyController,
                 'dpsk': MiuKlausController,
                 'time': MiuTemporalController,
                 'time2': MiuBoundedTemporalController}

    try:
        dpsclass = miu2class[args.miu]
    except:
        """
        try:
            smax_rate = re.search(r'(\d+)$', args.miu).group(0)
            miu_label = args.miu[:-len(smax_rate)]
            max_rate = int(smax_rate)
        except:
            miu_label = args.miu
            max_rate = 0
        """
        dpsclass = lambda *a, **kw: MiuRBFController(*a, n=args.rbfn, thres=args.thres,
                                                     signals=dpslab2signals(args.miu),
                                                     max_rate=args.maxrate,
                                                     miu_update_tstep=args.miustep,
                                                     miu_update_method=args.miuinterp,
                                                     **kw)
    return dpsclass


class MiuPolyController(MiuController):

    def __init__(self, dice, **kwargs):
        super().__init__(dice, vin=['X'], **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')

    def _body(self):
        self.u = self.new(RANGESET, 1, 5)
        self.X = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[-2,2], sow=0)
        self.TATM = self._dice.TATM #self.new(VAR, self._dice.t, doc='TATM <- DICE')
        self.MIU = self._dice.MIU #self.new(VAR, self._dice.t, doc='MIU -> DICE')
        self.MIU_bounds = self._dice.MIU_bounds  # self.new(VAR, self._dice.t, doc='MIU -> DICE')

    @staticmethod
    def eqmiu_rule(m, t):
        tm1 = max(1,t-1)
        tm2 = max(1,t-2)
        miu = np.maximum(
            np.minimum(
                m.MIU_bounds[t,1],
                (m.X[1] * m.TATM[tm1] +
                m.X[2] * (m.TATM[tm1])**2 +
                m.X[3] * (m.TATM[tm1] - m.TATM[tm2]) +
                m.X[4] * (m.TATM[tm1] - m.TATM[tm2])**2 +
                m.X[5] * m.TATM[tm1] * (m.TATM[tm1] - m.TATM[tm2]))),
            m.MIU_bounds[t,0])
        return miu


    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, MiuPolyController.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)


class MiuTemporalController(MiuController):

    def __init__(self, dice, **kwargs):
        super().__init__(dice, vin=['X'], **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')

    def _body(self):
        self.u = self.new(RANGESET, 1, len(self._dice.time._range)-1)
        self.X = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[0,1.2], sow=0)
        self.TATM = self._dice.TATM #self.new(VAR, self._dice.t, doc='TATM <- DICE')
        self.MIU = self._dice.MIU #self.new(VAR, self._dice.t, doc='MIU -> DICE')
        self.MIU_bounds = self._dice.MIU_bounds  # self.new(VAR, self._dice.t, doc='MIU -> DICE')

    @staticmethod
    def eqmiu_rule(m, t):
        if m.isfixed(m.MIU, t):
            return m.MIU[t]
        miu = np.maximum(
            np.minimum(
                m.MIU_bounds[t,1], m.X[t-1]),
            m.MIU_bounds[t,0])
        return miu
    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, MiuTemporalController.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)


class MiuBoundedTemporalController(MiuController):

    def __init__(self, dice, max_rate=4, **kwargs):
        self._default_max_rate = max_rate
        super().__init__(dice, vin=['X'], **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')

    def _body(self):
        self.u = self.new(RANGESET, 1, len(self._dice.time._range)-1)
        self.X = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds=[0,1.2], sow=0)
        self.max_rate = self.new(PARAM, doc='Maximum annual increase in MIU [% points]',
                                 default=self._default_max_rate, sow=0)
        self.TATM = self._dice.TATM #self.new(VAR, self._dice.t, doc='TATM <- DICE')
        self.MIU = self._dice.MIU #self.new(VAR, self._dice.t, doc='MIU -> DICE')
        self.MIU_bounds = self._dice.MIU_bounds  # self.new(VAR, self._dice.t, doc='MIU -> DICE')

    @staticmethod
    def eqmiu_rule(m, t):
        if m.isfixed(m.MIU, t):
            return m.MIU[t]
        miu = np.maximum(
            np.minimum(
                m.MIU_bounds[t,1],
                    m.X[t-1]),
            m.MIU_bounds[t,0])
        if m.max_rate > 0:
            assert t>1
            miu = np.maximum(
                np.minimum(
                    m.MIU[t-1]+m.max_rate/100.*m.tstep,
                    miu),
                np.minimum(1., m.MIU[t-1]))
        return miu

    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, MiuBoundedTemporalController.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)



class MiuKlausController(MiuController):

    def __init__(self, dice, **kwargs):
        super().__init__(dice, vin=['X'], **kwargs)
        self.set_outbound('MIU').set_inbound('TATM')
        self.hist_temp_data = get_hist_temp_data()
        self.nreg = 7
        self.nsows = kwargs['default_sow']
        self.hist_temp = (self.hist_temp_data.loc[(2010 - self.nreg * 5):2010:5] - (self.hist_temp_data.loc[2015] - 0.85)).values

    def _body(self):
        self.u = self.new(RANGESET, 1, 5)
        self.X = self.new(VAR, self.u, doc='MIU[t] = X*TATM[T]', bounds={1:[0,1], 2:[0,1], 3:[0,2], 4:[0,1], 5:[0,2]}, sow=0)
        self.TATM = self._dice.TATM #self.new(VAR, self._dice.t, doc='TATM <- DICE')
        self.MIU = self._dice.MIU #self.new(VAR, self._dice.t, doc='MIU -> DICE')
        self.MIU_bounds = self._dice.MIU_bounds  # self.new(VAR, self._dice.t, doc='MIU -> DICE')

    def eqmiu_rule(self, m, t):
        tm1 = max(1,t-1)
        tm2 = max(1,t-2)

        # Linear regression
        nreg = self.nreg # periods
        if t<=nreg:
            hist_temp = self.hist_temp[(t-nreg-1):]
            nsows = self.nsows
            hist_temp_sows = np.tile(hist_temp, [nsows, 1]).T
            obs_temp = np.vstack([hist_temp_sows, m.TATM[1:t]])
        else:
            obs_temp = m.TATM[t-nreg:t]
        obs_temp_mean = np.mean(obs_temp, 0)
        obs_time = np.arange(1, nreg + 1)
        obs_time_mean = (nreg + 1) / 2
        obs_time_squared = (obs_time - obs_time_mean) ** 2
        beta_num = np.dot((obs_temp - obs_temp_mean).T, (obs_time - obs_time_mean))
        beta_den = np.sum(obs_time_squared)
        beta = beta_num / beta_den
        alpha = obs_temp_mean - beta * obs_time_mean
        temp_proj50yrs = alpha + beta * (nreg + 10)

        miu = np.maximum(
            np.minimum(
                m.MIU_bounds[t,1],
                (m.X[1] * m.TATM[tm1] +
                 m.X[2] * m.MIU_bounds[t,1] * np.power(m.TATM[tm1]/2, m.X[3]) +
                 m.X[4] * m.MIU_bounds[t,1] * np.power(temp_proj50yrs/2, m.X[5])
                )),
            m.MIU_bounds[t,0])
        return miu

    def _body_eqs(self):
        # Write miu
        self.eqdicemiu = self.new(EQUATION, self.MIU, self.eqmiu_rule)
        # Wait for TATM
        self.eqdicetatm = self.new(EQUATION, self.TATM, None)
