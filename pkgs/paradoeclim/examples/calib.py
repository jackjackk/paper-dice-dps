from pymc3.backends.base import MultiTrace
from statsmodels.graphics.tsaplots import plot_acf

from paradoeclim import DoeclimCalib
from model import MODE_SIM, MODE_OPT
import matplotlib.pylab as plt
import logging
import dill
import theano.tensor as t
import theano
import pymc3 as pm
logging.basicConfig(level=logging.INFO)

vin = ['t2co', 'kappa', 'alpha', 'dq10', 'dq20', 'delql0', 'delqo0', 'dpast20', 'dteaux10', 'dteaux20']
dc = DoeclimCalib(name='doeclim_sim', mode=MODE_SIM, vin=vin)

calib = dc.run([6.595,33.572,0.803,4.790,-0.024,0.014,-0.014,-0.512,10.642,-3.414])
import seaborn as sb
plot_acf((calib.temp_year-dc.tempdata).dropna())
from borg4platypus import ExternalBorgC

#dc = DoeclimCalib(name='doeclim_sim', setup={'ns':{None:116}}, mode=MODE_OPT)

"""
# Fit a linear regression model to historical temp data
res=(dc.temp-dc.temphist)
import statsmodels.formula.api as s±mf
linfit = smf.OLS.from_formula(formula='tempdata ~ Year', data=dc.tempdata.reset_index())
r=linfit.fit()
fig,ax = plt.subplots(1,1)
dc.tempdata.plot(ax=ax)
linpred = r.predict(dc.tempdata.index.to_series())
linpred.plot(ax=ax)

# compare residual distribution w/ normal
import seaborn as sb
from scipy.stats.distributions import norm
import numpy as np
fig,ax = plt.subplots(1,1)
res = (linpred - dc.tempdata)
sb.kdeplot(res, ax=ax)
a = norm(res.mean(), res.std())
sb.kdeplot(a.rvs(10000), ax=ax)
x = np.linspace(*ax.get_xlim(),50)
ax.plot(x, norm.pdf(x))

# show residuals autocorrelation
plt.plot(res)
plot_acf(res)
ax.plot(dc.temp.values)
ax.plot(dc.temphist)
"""


# calibrate model
@theano.compile.ops.as_op(itypes=[t.dscalar]*len(vin),otypes=[t.dvector])
def run_doeclim(*args):
    d = dc.run(args)
    return d.temp.values

with pm.Model() as m:
    t2co = pm.Lognormal('t2co', mu=1.10704, sd=0.264, transform=None)
    kappa = pm.Uniform('kappa', 0.01, 100., transform=None)
    alpha = pm.Normal('alpha', mu=0, sd=1/3., transform=None)
    #offset = pm.Uniform('offset', -0.6, 0., transform=None)
    dq10 = pm.Normal('dq10', mu=0, sd=10/3.)
    dq20 = pm.Normal('dq20', mu=0, sd=1/3.)
    delql0 = pm.Normal('delql0', mu=0, sd=2/3.)
    delqo0 = pm.Normal('delqo0', mu=0, sd=2/3.)
    dpast20 = pm.Normal('dpast20', mu=0, sd=3/3.)
    dteaux10 = pm.Normal('dteaux10', mu=0, sd=15/3.)
    dteaux20 = pm.Normal('dteaux20', mu=0, sd=8/3.)
    mu = pm.Deterministic('mu', run_doeclim(t2co, kappa, alpha, dq10, dq20, delql0, delqo0, dpast20, dteaux10, dteaux20))
    sigma = pm.Uniform('sigma', 0.0001, 1., transform=None)
    y = pm.Normal('resid', mu=mu[:], sd=sigma, observed=dc.tempdata.loc[dc.d.year])
    #start = pm.find_MAP(maxeval=10)
    #step = pm.Metropolis()
    #step = pm.ADVI()
    #step = pm.Metropolis([t2co,kappa,alpha,sigma])
    #start={pname:pval for pname, pval in zip([t2co,kappa, alpha, sigma, offset],[3, 5, 0.3, 0.1, -0.25])}
    trace = pm.sample(draws=10000, step=pm.Metropolis(), njobs=4)

with open('trace_doeclim_calib.dat', 'wb') as f:
    dill.dump(trace, f)

with open('trace_doeclim_calib.dat', 'rb') as f:
    trace = dill.load(f)




"""
dc = DoeclimCalib(name='doeclim_sim', ns=116, mode=MODE_SIM, vin=['t2co','kappa','alpha'])
fig,ax = plt.subplots(1,1)
dc.tempdata[dc.year.value].plot(ax=ax)
for x in [[t['t2co'],t['kappa'],t['alpha']] for t in trace]:
    d = dc.run(x)
    d.temp.plot(ax=ax)
len(trace)

trace.get_values

"""
