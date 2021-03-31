from collections import namedtuple
import scipy.optimize

from dicedps.plot.common import *

inplot = lambda *x: os.path.join(os.environ['HOME'], 'working','meeting-keller-20180504','figures',*x)

#pprior = lambda x, y: plt.plot(x, y, ls='--', color='grey', zorder=0)  # **prop_list[2])
# if True: #any('(inf)' in x for x in args):
prior = ss.cauchy(loc=3, scale=2)
xcs = np.linspace(0.1, 10, 100)
h = prior.cdf(10) - prior.cdf(0.1)
ycs = (prior.pdf(xcs) / h).tolist()
#pprior(xcs, ycs)


a = 5.35
s2ecs = lambda x: np.array(x)*a*np.log(2)
q95 = [2.5, 97.5]
q90 = [5., 95.]
q80 = [10., 90.]
q68 = [16., 84.]
# https://www.nature.com/articles/nature11574

## Fig 3c
xs = np.array([2.5,
#               16., 84.,
               97.5])/100
ys = s2ecs(np.array([0.48,
#                     0.65, 1.27,
                     1.91]))
xmode = s2ecs(0.78)

PriorFit = namedtuple('PriorFit',['q','cs','dist','name'])


def densityfunc(*args, dist='lognorm', w='cdf'):
    if w in ['cdf','ppf','pdf']:
        ioff = 1
    else:
        ioff = 0
    if dist in ['lognorm',]:
        args = list(args[:ioff]) + [args[ioff],0] + list(args[(ioff+1):])
    if dist == 'truncnorm':
        myclip_a = 0.01
        myclip_b = 10.
        my_mean = args[ioff+0]
        my_std = args[ioff+1]
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        args = list(args[:ioff])+[a,b]+list(args[ioff:])
    return getattr(getattr(stats, dist), w)(*args)


def errfunc(x, ys=None, xs=None, fcdf=None, fppf=None):
    err = np.sum(np.square(xs - fcdf(ys, *x)))
    #err = np.sum(np.square(ys - fppf(xs, *x)))
          #np.square(xmode - np.exp(x[0] - x[1]**2))
    return err

dist2nargs = {'lognorm': 2}
fig, axs = plt.subplots(1,3, figsize=(12,6))
ax=axs[0]
xcs = np.linspace(0.1, 10, 100)
pf2pdf = {}
for pf in [
    #PriorFit(q=[2.5,16,84,97.5], cs=s2ecs([0.48, 0.65, 1.27, 1.91]), dist='lognorm', name='Paleosens 2012'),
    PriorFit(q=[2.5,97.5], cs=s2ecs([0.48, 1.91]), dist='lognorm', name='Lognorm ~ Paleosens 2012'),
    PriorFit(q=[2.5, 97.5], cs=[1.3, 2.3], dist='truncnorm', name='Truncnorm ~ Chylek & Lohmann 2008'),
    PriorFit(q=[17, 83], cs=[1.3, 2.3], dist='truncnorm', name='Truncnorm ~ Chylek & Lohmann 2008 (wide)'),
    #PriorFit(q=[16,50,84], cs=[2.2, 3.5, 4.8], dist='truncnorm'),
    #PriorFit(q=[10,50,90], cs=[1., 3.5, 6.], dist='truncnorm'),
    #PriorFit(q=[5,50,95], cs=[.55, 2.75, 7.95], dist='truncnorm'),
    #PriorFit(q=[10,90], cs=[1., 6.], dist='cauchy', name='IPCC'),
    ]:
    fcdf = partial(densityfunc, dist=pf.dist, w='cdf')
    ferr = partial(errfunc, xs=np.array(pf.q) / 100., ys=pf.cs, fcdf=fcdf)
    #fppf = partial(densityfunc, dist=pf.dist, w='ppf')
    #ferr = partial(errfunc, xs=np.array(pf.q) / 100., ys=pf.cs, fppf=fppf)
    xopt = scipy.optimize.fmin(ferr, x0=[.8]*dist2nargs.get(pf.dist, 2), maxiter=1000, maxfun=1000,
                               xtol=1e-8, ftol=1e-8)
    print(pf, xopt)
    xopt = np.round(xopt, 4)
    #fpdf = partial(densityfunc, dist=pf.dist, w='pdf')
    #ax.plot(xcs, densityfunc(xcs, *xopt, dist=pf.dist, w='pdf'), label=f'{str(pf)}\n{xopt} -> mean = {densityfunc(*xopt, dist=pf.dist,w="mean")}')
    kws = {}
    if pf.name == 'Truncnorm ~ Chylek & Lohmann 2008':
        kws = {'color':'grey','ls':'--'}
    ax.plot(xcs, densityfunc(xcs, *xopt, dist=pf.dist, w='pdf'), label=pf.name, **kws)

ax.plot(xcs, ss.cauchy.pdf(xcs, loc=3, scale=2), label='Cauchy ~ IPCC paleo range')
ss.cauchy.cdf(6, loc=3, scale=2)
ax.set_ylabel('PDF')
ax.set_xlabel('Climate sensitivity [K]')
ax.legend()

ax=axs[1]
pprior = lambda ax, x, y, **kws: ax.plot(x, y, **kws) # **prop_list[2])
puniprior = lambda ax, lo, up, **kws: pprior(ax, [lo, lo, up, up], [0, ] + [1 / (up - lo)] * 2 + [0, ], **kws)
puniprior(ax, 0, 4, label='OD unif in [0.1,4]')
puniprior(ax, 0, 10, label='OD unif in [0.1,10]')
ax.set_xlabel('Ocean diffusivity [cm^2/s]')
ax.legend()

ax=axs[2]
puniprior(ax, 0, 2, label='AS unif in [0,2]')
ax.legend()
ax.set_xlabel('Aerosol scaling')
fig.tight_layout()
fig.savefig(inplot('mypriors.pdf'))

prior = ss.cauchy(loc=3, scale=2)
xcs = np.linspace(0.1, 10, 100).tolist()
h = prior.cdf(10) - prior.cdf(0.1)
ycs = (prior.pdf(xcs) / h).tolist()

ax=axs[0]


xopt
densityfunc(xcs, *xopt, dist='lognorm', w='pdf')
plt.plot(pf2pdf['Chylek & Lohmann 2008 wide'](xcs))
pf2pdf['Chylek & Lohmann 2008 wide'](2)
## Abstract
xs = np.array([2.5, 97.5])/100
xs = np.array([16, 84])/100
#ys = s2ecs(np.array([0.6, 1.3]))
ys = np.array([2.2, 4.8])

# https://link.springer.com/article/10.1007%2Fs00382-017-3744-4
xs = np.array([10.,90.])/100
ys = np.array([1.,6.])

ss.lognorm.pdf(2, 1.2672, 0, 0.3523)

stats.cauchy.cdf
my_mean = 1.7997
my_std = 0.5242
myclip_a = 0.01
myclip_b = 10.
a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
ss.truncnorm.pdf(2, a, b, my_mean, my_std)


xopt

x0 = np.round(xopt[0], 4)
x1 = np.round(xopt[1], 4)
x0,x1
for (x0,x1), lab, p in zip([(1.1785, 0.3923), (1.2672, 0.3523), (0.8959, 0.6991)],
    ['abstract', 'fig3c', 'lewis 2017'], prop_list):
    ln = ss.lognorm(scale=np.exp(x0), s=x1)
    ax.plot(xcs, ln.pdf(xcs), label=lab, **p)
ax.legend()
ln.ppf(xs)/a/np.log(2)
ys/a/np.log(2)
x0
1.70-0.82
0.82-0.28
ss.norm(scale=0.82, s=1.23)