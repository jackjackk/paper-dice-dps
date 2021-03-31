import numpy as np
from collections import defaultdict
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger('misc')

# Recursive defaultdict
from pyomo.core.base import Expression

recurdict = lambda : defaultdict(recurdict)

def reorder_dims(darray, dim1, dim2):
    """
    Interchange two dimensions of a DataArray in a similar way as numpy's swap_axes
    """
    dims = list(darray.dims)
    assert set([dim1,dim2]).issubset(dims), 'dim1 and dim2 must be existing dimensions in darray'
    ind1, ind2 = dims.index(dim1), dims.index(dim2)
    dims[ind2], dims[ind1] = dims[ind1], dims[ind2]
    return darray.transpose(*dims)


def npv(xda, r=5., tfirst=2015, tlast=2200):
    y = xda.to_series().unstack('t').T
    y.index = y.index.to_series().apply(lambda x: 2015+(x-1)*5)
    t = range(tfirst, tlast+1)
    y = y.reindex(t).interpolate().mul(pd.Series([1/((1.+r/100.)**(x-2015.)) for x in t], index=t),0).sum()
    return xr.DataArray.from_series(y)


"""
def npv(xda, r=5, tfirst=1, tlast=17):
    tlen = tlast - tfirst + 1
    yrlen = (tlen - 1) * 5 + 1
    a = np.zeros([tlen, yrlen])
    i, j = np.indices(a.shape)
    a[j == i * 5] = 1.
    for x in range(1, 5):
        a[j == i * 5 + x] = (5 - x) / 5.
        a[j == i * 5 - x] = (5 - x) / 5.
    b = a.dot(np.power(1 / (1.+r/100.), np.arange(0, yrlen)))
    xdat = reorder_dims(xda.sel(t=slice(tfirst, tlast)),xda.dims[-1],'t')
    def dotprod(y, axis):
        return y.dot(b)
    return xdat.reduce(dotprod, 't')
"""

def cum2100(xda):
    return npv(xda, 0, 2015, 2100)

def npv2(xda):
    return ((5 * xda) * np.power(1 / 1.05, xda.sel(var='year', scen='bau', drop=True) - 2015)).sum('t')



def erf2(x):
    return 1-ex.exp(-16/23*x*x-2/sqrt(pi)*x)

def erf1(x):
    p = 0.3275911
    a = [0.254829592,
        -0.284496736,
         1.421413741,
        -1.453152027,
         1.061405429]
    t = 1 / (1 + p * x)
    return sum([a[i-1]*pow(t, i) for i in range(1,6)])*ex.exp(-x*x)
    #tau = t * ex.exp(-pow(x, 2)) - 1.26551223 + 1.00002368 * t + 0.37409196 * pow(t, 2) + 0.09678418 * pow(t, 3)
    #              - 0.18628806 * pow(t, 4) + 0.27886807 * pow(t, 5) - 1.13520398 * pow(t, 6)
    #              + 1.48851587 * pow(t, 7) - 0.82215223 * pow(t, 8) + 0.17087277 * pow(t, 9))
    #return 1 - tau

def erf(m,x):
    t = 1 / (1 + 0.5 * x)
    tau = t * m.exp(-pow(x, 2) - 1.26551223 + 1.00002368 * t + 0.37409196 * pow(t, 2) + 0.09678418 * pow(t, 3)
                  - 0.18628806 * pow(t, 4) + 0.27886807 * pow(t, 5) - 1.13520398 * pow(t, 6)
                  + 1.48851587 * pow(t, 7) - 0.82215223 * pow(t, 8) + 0.17087277 * pow(t, 9))
    return 1 - tau

def summation(v, idx):
    return np.sum(v[idx])

def rule_augmenter(prevrule, rule, force=True):
    def newrule(m, *args):
        if prevrule is None:
            x = None
        else:
            x = prevrule(m, *args)
        y = rule(m, *args)
        if y is Expression.Skip:
            return x
        if (not force) and (x is not None):
            return x
        return y
    return newrule


# Priority queue (Beazly)
import heapq
class CutPriorityQueue:
    def __init__(self, thres):
        self._queue = []
        self._index = 0
        self._thres = thres
        self._set = set()

    def push(self, item, priority):
        if item in self._set:
            logger.debug('Item {item} already in Queue'.format(item=item))
            return
        if (-priority)>self._thres:
            logger.debug('Item {item} has priority ({pri}) above threshold ({thres}), discarding'.format(item=item,pri=-priority,thres=self._thres))
            return
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1
        self._set.add(item)

    def pop(self):
        item = heapq.heappop(self._queue)[-1]
        self._set.remove(item)
        return item

    def __len__(self):
        return len(self._queue)


# Stopwatch

import time

class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        logger.info('Elapsed: {elapsed}'.format(elapsed=self.elapsed))
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
