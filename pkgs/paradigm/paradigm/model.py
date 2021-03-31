import dill as pickle
import datetime
from collections import defaultdict
from functools import partial
import itertools

from pandas import DataFrame
from pathos.multiprocessing import ProcessingPool
from platypus import Problem, Real
from pyomo.core.base import Constraint, AbstractModel, Suffix, Expression, Param, Var, RangeSet, Objective, Set, \
    minimize, maximize
from pyomo.core.util import summation
import pyomo.core.expr as ex
import numpy as np
import logging

from xarray import DataArray

from paradigm import misc
from .misc import rule_augmenter
from .pyomo import PyomoSolver
import os
from queue import Queue
import pandas as pd
import operator
import itertools as it
import pprint
logger = logging.getLogger(__name__)


def mybroadcast(x, tgt):
    if not hasattr(x, 'shape'):
        x = np.array(x)
    if x.ndim == 0:
        return x
    if tgt.shape == x.shape:
        return x
    assert tgt.ndim == x.ndim + 1
    return np.repeat(np.expand_dims(x, tgt.ndim), tgt.shape[-1], -1)


class SimSet(np.ndarray):

    EXTRA_ATTRS = ['istime','kwargs','name','ifirst']

    def __new__(subtype, *args, dtype=float, buffer=None, offset=0,
                strides=None, order=None, name=None, **kwargs):
        if len(args) == 2:
            # Constructor giving initial and last element of a range
            values = range(0, int(args[1]+args[0]))
            ifirst = args[0]
        else:
            # Constructor giving list of values
            values = kwargs.pop('initialize')
            ifirst = 0
        obj = np.asarray(values).view(subtype)
        obj.istime = kwargs.get('doc','').lower()[:4] == 'time'
        obj.kwargs = kwargs
        obj.name = name
        obj.ifirst = ifirst
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for prop in SimSet.EXTRA_ATTRS:
            setattr(self, prop, getattr(obj, prop, None))

    def __reduce__(self):
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + tuple(getattr(self, a) for a in SimSet.EXTRA_ATTRS)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        for a, val in zip(SimSet.EXTRA_ATTRS, state[-len(SimSet.EXTRA_ATTRS):]):
            setattr(self, a, val)
        super().__setstate__(state[:-len(SimSet.EXTRA_ATTRS)])

    @property
    def value(self):
        return self[self.ifirst:]


class SimData(np.ndarray):

    EXTRA_ATTRS = ['isdynamic', 'name', 'initialize',
                   'fixed', 'bounds', 'parallel','domain','doc',
                   # 'domain',
                   # 'kwargs',
        ]

    def __new__(subtype, *args, dtype=float, buffer=None, offset=0,
                strides=None, order=None, name=None, bounds=None, parallel=0, doc=None, **kwargs):
        #for x in args:
        #    assert isinstance(x, SimSet)
        default = kwargs.pop('default', None)
        if len(args) == 0:
            shape = default
            dim = 0
        else:
            shape = [len(x) for x in args]
            dim = len(args)
        if isinstance(shape,list):
            _obj = np.full(shape, np.nan, dtype=dtype)
            if default is not None:
                if isinstance(default, dict):
                    for k,v in default.items():
                        _obj[k,:] = v
                else:
                    _obj[:] = mybroadcast(default, _obj)
        else:
            _obj = np.array(shape, dtype=dtype)
        obj = _obj.view(subtype)
        if isinstance(shape,list):
            _fixed = np.zeros(shape, dtype=bool)
            # Fix the very first element across all dimensions
            for d in range(dim):
                _fixed[tuple([slice(0,args[j].ifirst) if j==d else slice(None,None) for j in range(dim)])] = True
        else:
            _fixed = np.array(0, dtype=bool)
        obj.fixed = _fixed
        if dim>0 and args[0].istime:
            obj.isdynamic = True
        else:
            obj.isdynamic = False
        obj.domain = args
        obj.bounds = bounds
        obj.doc = doc
        #obj.kwargs = kwargs
        obj.name = name
        obj.parallel = parallel
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for prop in SimData.EXTRA_ATTRS:
            setattr(self, prop, getattr(obj, prop, None))

    def __reduce__(self):
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + tuple(getattr(self, a) for a in SimData.EXTRA_ATTRS)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        for a, val in zip(SimData.EXTRA_ATTRS, state[-len(SimData.EXTRA_ATTRS):]):
            setattr(self, a, val)
        super().__setstate__(state[:-len(SimData.EXTRA_ATTRS)])

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except:
            if isinstance(item, str):
                item = [item]
            elif isinstance(item, tuple):
                item = list(item)
            for iid, ival in enumerate(item):
                item[iid] = np.where(self.domain[iid] == ival)[0][0]
            if len(item) == 1:
                item = item[0]
            else:
                item = tuple(item)
            return super().__getitem__(item)

    def dim(self):
        return self.ndim

    @property
    def value(self):
        if self.ndim == 0:
            return self
        return self[tuple([slice(x.ifirst, None) for x in self.domain])]


class SimEquation(object):

    def logf(self, f):
        def flogged(*args):
            logger.debug('{name} : {target} ({args})'.format(name=self.name, target=self.target.name, args=args))
            return f(*args)
        return flogged

    @staticmethod
    def placeholder_func(*args):
        return np.nan

    def toggle(self):
        if (self.eval == SimEquation.placeholder_func) and (self._eval_bak == SimEquation.placeholder_func):
            raise Exception('Trying to toggle a placeholder equation twice ({name})!'.format(name=self.name))
        self.eval, self._eval_bak = self._eval_bak, self.eval

    def __init__(self, target, rhs, name=None, parallel=0, vectorize=False, log=False, **kwargs):
        self.target = target
        self.name = name
        self.isdynamic = target.isdynamic
        self.parallel = parallel
        self.eval_just_once = kwargs.get('eval_just_once', False)
        self._eval_bak = SimEquation.placeholder_func
        if rhs is None:
            self.eval = None  # SimEquation.placeholder_func
        else:
            if log:
                rhs = self.logf(rhs)
            self.f = rhs
            if vectorize:  # TODO
                tgt_shape = self.target.shape
                self._f = np.vectorize(rhs)
                self._idx = np.indices(tgt_shape[:len(tgt_shape)-parallel])
                self.parallel = len(tgt_shape)
                self.f = lambda m: self._f(m, self._idx).T
            self.eval = self._eval_recur

    """
    def parallelize_f(self, m):
        self.f = np.vectorize(partial(self.f, m))
        assert self.parallel
        self.eval = self._eval_vector
        self.idx = np.indices(self.target.shape)"""

    def _eval_recur(self, m, *args):
        y = self.target[args]
        if y.ndim <= self.parallel:
            assert not m.isfixed(self.target, args)
            ret = self.f(m, *args)
            if ret is not None:
                try:
                    self.target[args] = ret
                except:
                    self.target[args] = mybroadcast(ret, self.target[args])
            return ret
        for x in range(1, y.shape[0]):
            ret = self._eval_recur(m, *args, x)

    def _eval_vector(self, m, *args):
        self.target[:] = self.f(m, self.idx)


    """
    def _nestedsimrule_generator(self, name, itup, minlen4nested_iter, simrule):
        # Multi dim -> iterate over other dimensions
        def _nestedsimrule(m, *args):
            _name = name
            created_atleast_once = False
            for j in it.product(*[x.elems for x in itup[(minlen4nested_iter - 1):]]):
                rhs = simrule(m, *args, *j)
                if (rhs != Constraint.Skip):
                    created_atleast_once = True
                elif created_atleast_once:
                    break

        return _nestedsimrule
    """


class SimObjective(object):

    def _wrap_sow_reduce(self, f, sow_reduce):
        def fwrapped(m, *args):
            ret = f(m, *args)
            if isinstance(ret, float):
                ret = np.array([ret])
            if ret.ndim>0:
                ret = sow_reduce(ret)
            return ret
        return fwrapped

    def __init__(self, name=None, sense=minimize, **kwargs):
        pyomo2platypus = {
            minimize: Problem.MINIMIZE,
            maximize: Problem.MAXIMIZE
        }
        self._kwargs = kwargs
        self._name = name
        self.sense = pyomo2platypus[sense]
        self.active = True
        self.value = SimData()
        rhs = self._wrap_sow_reduce(kwargs.pop('rule'), kwargs.pop('sow_reduce', sum))
        self.equ = SimEquation(self.value, rhs, name='{name}eq'.format(name=name), **kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.equ.name = '{name}eq'.format(name=value)

    def eval(self, m):
        self.equ.eval(m)
        return self.value

    def deactivate(self):
        self.active = False
        pass


# TODO: optimize code, now clone of SimObjective
# TODO: extend support for Brushes other than <= 0

class SimBrush(object):

    def _wrap_sow_reduce(self, f, sow_reduce):
        def fwrapped(m, *args):
            ret = f(m, *args)
            if isinstance(ret, float):
                ret = np.array([ret])
            if ret.ndim>0:
                ret = sow_reduce(ret)
            return ret
        return fwrapped

    def __init__(self, name=None, sense=minimize, **kwargs):
        pyomo2platypus = {
            minimize: Problem.MINIMIZE,
            maximize: Problem.MAXIMIZE
        }
        self._kwargs = kwargs
        self._name = name
        self.sense = pyomo2platypus[sense]
        self.active = True
        self.value = SimData()
        rhs = self._wrap_sow_reduce(kwargs.pop('rule'), kwargs.pop('sow_reduce', sum))
        self.equ = SimEquation(self.value, rhs, name='{name}eq'.format(name=name), **kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.equ.name = '{name}eq'.format(name=value)

    def eval(self, m):
        self.equ.eval(m)
        return self.value

    def deactivate(self):
        self.active = False
        pass


class Equation(Constraint):

    def _eqrule_generator(self, name, rhs, op=operator.eq):
        def eqrule(m, *args):
            _rhs = rhs(m, *args)
            if _rhs is None:
                return Constraint.Skip
            if len(args) == 0:
                return op(getattr(m, name), _rhs)
            return op(getattr(m, name)[args], _rhs)
        return eqrule

    def __init__(self, v, rhs, op=operator.eq, **kwargs):
        assert 'rule' not in kwargs
        name = v.name
        eqrule = self._eqrule_generator(name, rhs, op=op)
        #simrule = self._simrule_generator(name, rhs)
        super().__init__(v._index, rule=eqrule, **kwargs)
        #self._simrule = simrule


class Limit(Equation):

    def __init__(self, *args, op=operator.le, **kwargs):
        super().__init__(*args, op=op, **kwargs)


class SimRule(object):

    def __init__(self, compname):
        self.compname = compname
        self.eval_just_once = True
        self.isdynamic = False

    def eval(self, m):
        cname = self.compname
        comp = getattr(m, cname)
        input_values = m.setup[cname]
        try:
            input_values = float(input_values)
            if comp.ndim == 0:
                input_values = {None: input_values}
            else:
                input_values = {int(x): input_values for x in np.nditer(np.ma.where(~(comp.fixed)))}
        except:
            n_values_provided = len(input_values)
            if not isinstance(input_values, dict):
                input_values = {int(x):y for x,y in zip(np.nditer(np.ma.where(~(comp.fixed))), input_values)}
                n_values_set = len(input_values)
                if n_values_provided != n_values_set:
                    logger.info('{cname} rule: {n_values_provided} values provided,'
                                ' {n_values_set} values set'.format(cname=cname,
                                                                    n_values_provided=n_values_provided,
                                                                    n_values_set=n_values_set))
        for k, v in input_values.items():
            comp[k] = v
            try:
                m.setfixed(comp, k)
            except:
                pass


class SimFunction(object):

    def __init__(self, f):
        self.eval_just_once = True
        self.isdynamic = False
        self.f = f

    def eval(self, m):
        self.f(m)


class Simulator(object):

    def __init__(self, name, vin=[], vout=None, con=None, run_init=False):
        super().__init__()
        self._name = name
        self._vin = vin
        self._vinlen = [(None, 1)]*len(vin)
        self._eqs_time = []
        self._eqs_notime = []
        self._objs = []
        self._constrs = []
        self._v2eq = defaultdict(list)
        self._v2setup = {}
        self._run_init = run_init
        self._setlubounds = SimSet(0, 2)
        self._setlubounds.name = 'bounds'
        self.d = Data(self)
        self._symbols = []
        self._set_by_other_model = set()
        self._given_to_other_model = []
        self.clear_collateral_outputs()
        self._vout = vout
        self._con = con
        if vout is None:
            self.add_obj_if = lambda o: o.active
        else:
            self.add_obj_if = lambda o: o.name in vout
        if con is None:
            self.add_constr_if = lambda c: c.active
        else:
            self.add_constr_if = lambda c: c.name in con

    def _after_init(self):
        self._active_objs = [o for o in self._objs if self.add_obj_if(o)]
        self._active_constrs = [c for c in self._constrs if self.add_constr_if(c)]

    def set_inbound(self, *args):
        assert len(self._v2eq) > 0, 'Equations not initialized yet: cannot set inbounds'
        already_set = self._set_by_other_model
        toset_set = set(args)
        all_set = toset_set.union(already_set)
        process_set = all_set - toset_set.intersection(already_set)
        for v in process_set:
            eqlist = self._v2eq[v]
            if len(eqlist) == 0:
                eqlist = self._register_target_equation(SimEquation(getattr(self, v), None, 'eq{v}_placeholder'.format(v=v.lower())))
            elif len(eqlist) > 1:
                raise Exception('Many equations associated with target: which one to toggle?')
            eqlist[0].toggle()
        self._set_by_other_model = toset_set
        return self

    def set_outbound(self, *args):
        self._given_to_other_model = args
        return self

    def clear_collateral_outputs(self):
        self._collateral_outputs = []

    def get_collateral_outputs(self):
        return self._collateral_outputs

    def add_collateral_output(self, yr, varname):
        self._collateral_outputs.append((yr, varname))

    def _register_target_equation(self, value, index=None):
        if value.isdynamic:
            tgtlist = self._eqs_time
        else:
            tgtlist = self._eqs_notime
        if index is None:
            index = len(tgtlist)
        tgtlist.insert(index, value)
        eqlist = self._v2eq[value.target.name]
        eqlist.append(value)
        return eqlist

    def __setattr__(self, name, value):
        if name[0] == '_':
            return super().__setattr__(name, value)
        if isinstance(value, SimSet):
            if value.istime:
                self._t = value[1:]
        elif isinstance(value, SimConstraint):
            logger.info('Skipping constraint "{name}", not supported yet'.format(name=name))
            return
        elif isinstance(value, SimEquation) or isinstance(value, SimRule) or isinstance(value, SimFunction):
            if value.eval_just_once:
                value.eval(self)
            else:
                self._register_target_equation(value)
            # TODO: if value.parallel:
            #    value.parallelize_f(self)
        elif isinstance(value, SimData):
            pass
        elif isinstance(value, SimObjective):
            self._objs.append(value)
        elif isinstance(value, SimBrush):
            self._constrs.append(value)
        #else:
        #    raise Exception('not yet implemented')
        try:
            value.name = name
        except:
            pass
        super().__setattr__(name, value)

    def get_bounds(self, vin=None):
        bounds = []
        j = 0
        if vin is None:
            vin = self._vin
        else:
            vin = [vin]
        for v in vin:
            vobj = getattr(self, v)
            vbounds = getattr(self, '{v}_bounds'.format(v=v))
            for i in np.nditer(np.ma.where(~(vobj.fixed))):
                if vobj.ndim==0:
                    currbnds = vbounds
                else:
                    currbnds = vbounds[i]
                bounds.append(currbnds)
                logger.debug('x[{j}] -> {v}[{i}] within {currbnds}'.format(j=j,v=v,i=i,currbnds=currbnds))
                j += 1
        return bounds

    def run(self, *args, **kwargs):
        step = self.step(*args, **kwargs)
        try:
            while True:
                next(step)
        except StopIteration:
            pass
        return self

    def set_x(self, *args):
        x=np.array(args).flat
        j = 0  # counter for x
        for v in self._vin:
            #jstart = j
            vobj = getattr(self, v)
            for i in np.nditer(np.ma.where(~(vobj.fixed))):
                if vobj.ndim>0:
                    vobj[i] = x[j]
                else:
                    vobj[None] = x[j]
                j += 1
            #logger.debug(f'x[{jstart:3d}:{(j-1):3d}] -> {v}')
        assert j == len(x), '{xlen} variables given, {j} variables assigned'.format(xlen=len(x),j=j)

    def step(self, *args, time=None, eval_eqs_notime=True, **kwargs):
        if len(kwargs) > 0:
            args = list(itertools.chain(*[kwargs[v] for v in self._vin]))
        if len(args)>0:
            self.set_x(*args)
        """
            for k, v in kwargs.items():
                if isinstance(v, list):
                    getattr(self, k)[:] = v
                else:
                    assert False
        """

        if time is None:
            time = self.time
            offset = 1
        else:
            assert time.tstep == self.time.tstep
            assert time.start >= self.time.start
            assert time.end <= self.time.end
            offset = time.start - self.time.start + 1

        if eval_eqs_notime:
            for e in self._eqs_notime:
                if e is None:
                    continue
                e.eval(self)

        yield_at_end_of_timestep = (len(self._set_by_other_model) == 0) and (self._given_to_other_model == 0)
        for t, yr in enumerate(time._range):
            t += offset
            yr = yr.year
            for e in self._eqs_time:
                if e is None:
                    continue
                if self.isfixed(e.target, t):
                    ret = e.target[t]
                else:
                    ret = e.eval(self, t)
                if e.target.name in self._set_by_other_model:
                    yield self.get_collateral_outputs(), (yr, e.target.name)
                    self.clear_collateral_outputs()
                elif (isinstance(ret, float) and np.isnan(ret)) or \
                     (isinstance(ret, np.ndarray) and np.isnan(ret).any()):
                    raise Exception('There are nan\'s in the system ({etargetname}, {ename})!'.format(etargetname=e.target.name,ename=e.name))
                elif e.target.name in self._given_to_other_model:
                    self.add_collateral_output(yr, e.target.name)
                elif ret is None:
                    continue
            if yield_at_end_of_timestep:
                yield self.get_collateral_outputs(), None
                self.clear_collateral_outputs()

        for o in self._objs:
            o.eval(self)
        for c in self._constrs:
            c.eval(self)

        yield self.get_collateral_outputs(), None
        self.clear_collateral_outputs()

    def minimum(self, *args):
        return np.minimum(*args)

    def setfixed(self, v, idx, setval=True):
        if isinstance(v, str):
            v = getattr(self, v)
        # assert isinstance(v, SimData)
        v.fixed[idx] = setval

    def isfixed(self, v, idx):
        if isinstance(v, str):
            v = getattr(self, v)
        return v.fixed[idx].all()

    def getvalue(self, v, idx=None):
        if idx is None:
            return v
        try:
            return v[idx]
        except:
            return v

    def add_rule(self, varname, rule, force=True):
        try:
            tgtlist, idx = self._v2eq[varname]
            prevrule = tgtlist[idx]
            if prevrule is None:
                tgtlist[idx] = SimEquation(getattr(self, varname), rule)
            else:
                newrule = rule_augmenter(prevrule.f, rule)
                tgtlist[idx].f = newrule
        except:
            setattr(self, '{varname}eq'.format(varname=varname), SimEquation(getattr(self, varname), rule, eval_just_once=True))
        return self

    def summation(self, v):
        return sum(v[t] for t in self.t if ((self.year[t]>=self.baseyear) and (self.year[t])<=self.endyear))

    def run_and_ret_objs_list(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return [float(o.value) for o in self._active_objs]

    def run_and_ret_objs_constrs_list(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return ([float(o.value) for o in self._active_objs],
                [float(c.value) for c in self._active_constrs])

    def run_and_ret_objs(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return {o.name: float(o.value) for o in self._active_objs}

    def asproblem(self):
        active_constrs = self._active_constrs
        nconstrs = len(active_constrs)
        if nconstrs == 0:
            f = self.run_and_ret_objs_list
        else:
            f = self.run_and_ret_objs_constrs_list
        active_objs = self._active_objs
        nobjs = len(active_objs)
        bounds_list = self.get_bounds()
        nvars = len(bounds_list)
        p = Problem(nvars, nobjs, nconstrs=nconstrs, function=f)
        p.types[:] = [Real(bnds[0], bnds[1]) for bnds in bounds_list]
        p.constraints[:] = "<=0"
        p.directions[:] = [o.sense for o in active_objs]
        return p

    def asmodel(self):
        import rhodium.model as rhm
        #active_objs = self._active_objs
        platypus2rhodium = {Problem.MINIMIZE: rhm.Response.MINIMIZE,
                            Problem.MAXIMIZE: rhm.Response.MAXIMIZE}
        model = rhm.Model(self.run_and_ret_objs)

        model.parameters = [rhm.Parameter(v) for v in self._vin]

        model.responses = [rhm.Response(o.name, platypus2rhodium[o.sense]) for o in self._objs if self.add_obj_if(o)]

        # Define any constraints (can reference any parameter or response by name)
        # model.constraints = [Constraint("reliability >= 0.95")]

        # Some parameters are levers that we control via our policy
        levers = []
        for v in self._vin:
            bnds = self.get_bounds(v)
            levers.append(rhm.RealLever(v, min(b[0] for b in bnds), max(b[1] for b in bnds), length=len(bnds)))
        model.levers = levers

        model.uncertainties = []
        return model

    def bounder(self, varname, varobj):
        def fbound(m, *args):
            try:
                i = m._vin.index(varname)
                self._vbounds[i] = SimData(varobj.domain+[SimSet(0,2),], parallel=varobj.parallel+1)
                if callable(varobj.bounds):
                    SimEquation(self._vbounds[i], varobj.bounds).eval(self)
                else:
                    self._vbounds[i] = varobj.bounds
            except:
                if varobj.bounds != (None, None):
                    logger.debug('Ignoring bounds for {varname}: {varobjbounds}'.format(varname=varname,varobjbounds=varobj.bounds))
                pass
        return fbound


class SimConstraint(SimEquation):

    def __init__(self, *args, op=operator.le, **kwargs):
        self.op = op
        super().__init__(*args, **kwargs)

    def _eval_recur(self, m, *args):
        y = self.target[args]
        if y.ndim <= self.parallel:
            ret = self.f(m, *args)
            if ret is not None:
                ret = self.op(self.target[args], ret)
            return ret
        for x in range(1, y.shape[0]):
            ret = self._eval_recur(m, *args, x)

    def _eval_vector(self, m, *args):
        return self.op(self.target[:], self.f(m, self.idx))


class ConfigurableModel(AbstractModel):

    def _after_init(self):
        pass

    def _init_scalars(self):
        pass

    def _init_sets(self):
        pass

    def _init_params(self):
        pass

    def _init_variables(self):
        pass

    def _init_equations(self):
        pass

    def _init_suffixes(self):
        # Solver Suffixes
        ## Ipopt bound multipliers (obtained from solution)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        ## Ipopt bound multipliers (sent to solver)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        ## Obtain dual solutions from first solve and send to warm start
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.data = {}

    def apply(self, f):
        f(self)
        self._applied_flist.append(f)
        logger.debug('')
        return self

    def apply_list(self, flist):
        for f in flist:
            self.apply(f)
        return self

    def setname(self, name):
        self.name = name
        return self

    def duplicate(self, setup={}, apply=[], rules={}, onlydata=False):
        _setup = self.setup
        _setup.update(setup)
        _apply = self._applied_flist
        _apply.extend(apply)
        _rules = self._rules
        _rules.update(rules)
        return type(self)(name=self.name, setup=_setup, apply=_apply, rules=_rules, onlydata=onlydata)

    def solve(self, **kwargs):
        i = self._default_solver(self, **kwargs)
        return i

    def run(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def __init__(self, name=None, rules={}, apply=[], onlydata=False):
        super().__init__(name=name)
        if name is None:
            name = '{classname}_{date}'.format(classname=self.__class__.__name__.lower(),date=datetime.datetime.now().strftime("%Y%m%d"))
        self.name = name
        #for k, v in setup.items():
        #    self.set(k, v)
        self._applied_flist = []
        self._rules = defaultdict(list)

        self.apply_list(apply)
        for k,v in rules:
            self.add_rule(k, v)

        self._init_suffixes()
        self._default_solver = PyomoSolver()
        self.d = Data(self)
        self._symbols = []

    @staticmethod
    def setup2pyomo(setup):
        pyomo_setup = {}
        for k,v in setup.items():
            try:
                for kk, vv in v.items():
                    break
            except:
                v = {None: v}
            pyomo_setup[k] = v
        return pyomo_setup

    def create_instance(self):
        setup = ConfigurableModel.setup2pyomo(self.setup)
        logger.debug('DATA:\n{data}\n'.format(data=pprint.pformat(setup, width=1, indent=4)))
        return super().create_instance(data={None:setup})

    def add_rule(self, varname, rule, force=True):
        self._rules[varname].append(rule)
        comp = getattr(self, varname)
        prevrule = comp._value_init_rule
        comp._value_init_rule = rule_augmenter(prevrule, rule, force)
        logger.debug('[{name}.{varname}] has new rule "{rule}"'.format(name=self.name,varname=varname,rule=rule))
        return self

    def setfixed(self, v, idx, setval=True):
        if isinstance(v, str):
            v = getattr(self, v)
        # assert isinstance(v, Var)
        if idx is not None:
            v[idx].fixed = setval
        else:
            v.fixed = setval
        return

    def minimum(self, *args):
        return min(*args)

    def isfixed(self, v, idx):
        if isinstance(v, str):
            v = getattr(self, v)
        return v[idx].fixed

    def getvalue(self, v, idx=None):
        try:
            return v[idx].value
        except:
            try:
                return v[idx]
            except:
                return v

    def summation(self, v):
        return summation(v)

    def compareto(self, m):
        self.pprint('tmp_left_{name}.txt'.format(name=self.name))
        m.pprint('tmp_right_{name}.txt'.format(name=m.name))
        os.system('bcompare tmp_left_{selfname}.txt tmp_right_{mname}.txt'.format(selfname=self.name,mname=m.name))


MODE_SIM, MODE_OPT = range(2)
NCOMP = 9
SET, RANGESET, PARAM, VAR, EQUATION, CONSTRAINT, OBJECTIVE, BRUSH, CALIBPARAM = range(NCOMP)


def wrap_pyomo_finit(f):
    def fwrapped(m, *i):
        if callable(f):
            ret = f(m, *i)
        else:
            ret = f
        if ret is None:
            return Expression.Skip
        return ret
    return fwrapped


def is_optimal(m):
    return (str(m.report['Solver.Termination condition']) == 'optimal') and \
            (str(m.report['Solver.Status']) == 'ok')


class Data(object):

    def __init__(self, m):
        self._m = m
        self._mode = MODE_SIM if isinstance(m, Simulator) else MODE_OPT
        self._cache = {}

    def __getattr__(self, name):
        if (name[0] == '_'):
            return super().__getattr__(name)
        index2year = False
        ret_only_freevars = False
        if '_year' in name:
            index2year = True
            name = name.replace('_year','')
        if '_free' in name:
            ret_only_freevars = True
            name = name.replace('_free','')
        try:
            ret = self._cache[name]
        except:
            v = getattr(self._m, name)
            if not hasattr(v, 'dim'):
                return v
            ndim = v.dim()
            daname = '{vname}'.format(vname=v.name)
            vdoc = getattr(v, 'doc', None)
            if vdoc is not None:
                daname += ': {vdoc}'.format(vdoc=vdoc)
            if ndim == 0:
                ret = DataArray(v.value, name=daname)
            else:
                if self._mode == MODE_SIM:
                    coord_objs = v.domain
                    data = v.value
                else:
                    if ndim == 1:
                        coord_objs = [v.index_set()]
                    else:
                        coord_objs = v.index_set().set_tuple
                coords = [x.value for x in coord_objs]
                dims = [x.name for x in coord_objs]
                if self._mode == MODE_SIM:
                    ret = DataArray(data, coords=coords, dims=dims, name=daname)
                else:
                    if isinstance(v, Param):
                        data = list(v.values())
                    else:
                        data = [x.value for x in v.values()]
                    if ndim == 1:
                        indices = [[i] for i in v.index_set()]
                    else:
                        indices = [i for i in v.index_set()]
                    df = DataFrame([[*i, d] for i,d in zip(indices, data)], columns=dims+[name,])
                    s = df.set_index(dims)[name]
                    ret = DataArray.from_series(s)
            ret = ret.to_pandas()
            self._cache[name] = ret
        if ret_only_freevars:
            ret = ret.copy()
            v = getattr(self._m, name)
            for sid, sval in ret.iteritems():
                if self._m.isfixed(v, sid):
                    ret.drop(sid, inplace=True)
        if index2year:
            ret = ret.rename(self.year)
        return ret

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self._cache[key] = value

    def keys(self):
        try:
            slist = self._m._symbols
        except:
            slist = self._cache.keys()
        return slist

    def save(self, name, savemodel=False):
        try:
            self._cache['report'] = self._m.report
            if not is_optimal(self._m):
                logger.warning('Not saving {name}: NOT OPTIMAL, report = \n{report}'.format(name=name,report=self._m.report))
                return
        except:
            logger.warning('No report found')
        for s in self.keys():
            getattr(self, s)
        if savemodel:
            self._cache['_m'] = self._m
        if name[-4:].lower() != '.dat':
            name += '.dat'
        path = name
        dirpath = os.path.dirname(path).strip()
        if (dirpath != '') and (not os.path.exists(dirpath)):
            logger.debug('"{dirpath}" not found, creating it'.format(dirpath=dirpath))
            os.makedirs(dirpath)
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        if savemodel:
            self._cache.pop('_m')
        logger.debug('Saved to "{path}"'.format(path=path))
        return self

    @staticmethod
    def load(name, fifnotfound=None, force_reload=False):
        """
        Load a Data instance from a file.

        :param name: filename to load [str]
        :param fifnotfound: function to execute if filename not found, in which case returned Data is saved to filename [Callable]
        :param force_reload: call fifnotfound anyway if True [bool]
        :return: Data instance
        """
        if name[-4:].lower() != '.dat':
            path = '{name}.dat'.format(name=name)
        else:
            path = name
        d = Data(None)
        try:
            assert not force_reload
            with open(path, 'rb') as f:
                d._cache = pickle.load(f)
                try:
                    d._m = d._cache.pop('_m')
                except:
                    logger.debug('No model instance found, only data')
                logger.info(f'Loaded data from {path} in {d}')
                try:
                    if 'Optimal Solution Found' not in d.report['Solver.Message']:
                        logger.warning('{name} optimal but message = "{report}"'.format(name=name,report=d.report["Solver.Message"]))
                except:
                    pass

        except:
            assert callable(fifnotfound), 'File not found, and no generator function provided!'
            logger.debug('Running function for "{name}"'.format(name=name))
            d = fifnotfound().save(name)
        return d

    def invalidate_cache(self):
        self._cache = {}


class Time(object):

    def __init__(self, start=None, end=None, periods=None, tstep=1, **kwargs):
        self._range = pd.period_range(start=start, end=end, periods=periods, freq='{tstep}A-DEC'.format(tstep=tstep), **kwargs)
        self._tstep = tstep

    @property
    def start(self):
        return self._range[0].year

    @property
    def end(self):
        return self._range[-1].year

    @property
    def len(self):
        return len(self._range)

    @property
    def tstep(self):
        return self._tstep

    def year(self, i):
        return self._range[i-1].year


class Model(object):
    MATH_FUNCS = ['exp','log','erf']
    mode2model = [Simulator, ConfigurableModel]
    mode2component = [None]*NCOMP
    mode2component[SET] = [SimSet, Set]
    mode2component[RANGESET] = [SimSet, RangeSet]
    mode2component[PARAM] = [SimData, Param]
    mode2component[VAR] = [SimData, Var]
    mode2component[EQUATION] = [SimEquation, Equation]
    mode2component[CONSTRAINT] = [SimConstraint, Limit]
    mode2component[OBJECTIVE] = [SimObjective, Objective]
    mode2component[BRUSH] = [SimBrush, Limit]
    mode2mathmodule = [np, ex]
    mode2kwargs2pop = [[],['dtype','only','parallel','vectorize','sow','eval_just_once','sow_reduce',
                           'vin','sow_setup']]
    calib2calibparam_comptype = [PARAM, VAR]

    def _after_solve(self, m):
        pass

    def _init_scalars(self):
        pass

    def _init_sets(self):
        pass

    def _init_params(self):
        pass

    def _init_variables(self):
        pass

    def _init_equations(self):
        pass

    def _body(self):
        pass

    def _body_eqs(self):
        pass

    def __init__(self, time=None, name=None, mode=MODE_OPT, setup={}, calib=False, default_sow=0, sow_setup={}, **kwargs):
        super().__init__()
        self._queue = []
        self._mode = mode
        self._name = name
        for k in Model.mode2kwargs2pop[mode]:
            try:
                kwargs.pop(k)
                logger.info('"{k}" kwarg not supported'.format(k=k))
            except:
                pass
        self._kwargs = kwargs
        self._model = Model.mode2model[mode](name, **kwargs)
        _setup = defaultdict(dict)
        _setup.update(setup)
        self.setup = _setup
        self._sow_setup = sow_setup
        self._math = Model.mode2mathmodule[mode]
        self._default_sow = default_sow
        self._calib = calib
        self._calibvcounter = 0
        self._idnum = 0
        if default_sow>0:
            assert self._mode == MODE_SIM, 'SOW indexing supported only in simulation mode'
            self.sow = self.new(RANGESET, 0, default_sow, doc='Set of states of the world')
        # Link correct version of mathematical functions
        for mfunc_name in Model.MATH_FUNCS:
            try:
                mfunc = getattr(self._math, mfunc_name)
            except:
                mfunc = partial(getattr(misc, mfunc_name), self._model)
            setattr(self._model, mfunc_name, mfunc)

        # Init time
        if time is not None:
            self._model.time = time
            self._model.baseyear = time.start
            self._model.endyear = time.end
            self._model.tstep = time.tstep
            self._model.dt = time.tstep
            self._model.tnum = time.len
            self._model.ns = time.len
            self.t = self.new(RANGESET, 1, time.len, doc='Time')
            self.year = self.new(PARAM, self.t, initialize=lambda m,i: time.year(i), dtype=int, sow=0)

        # Rest
        self._init_scalars()
        self._init_sets()
        self._init_params()
        self._init_variables()
        self._init_equations()
        self._body()
        self._body_eqs()

        self._after_init()

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        if name[0] == '_':
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)
            if hasattr(value, '_idnum'):
                if value._comptype in [PARAM, VAR]:
                    self._model._symbols.append(name)
                q = self._queue[value._idnum]
                while not q.empty():
                    setattr(self, *(q.get()(name, value)))
                del value._idnum

    def run(self, *args, **kwargs):
        self.d.invalidate_cache()
        ret = self._model.run(*args, **kwargs)
        return ret.d

    def set(self, *args, **kwargs):
        if len(args) > 0:
            assert len(args) == 2
            kwargs[args[0]] = args[1]
        for compname, val in kwargs.items():
            if not isinstance(compname, str):
                compname = compname.name
            if isinstance(val, float) or isinstance(val, int) or isinstance(val, list):
                val = {None: val}
            elif isinstance(val, np.ndarray):
                comp = getattr(self, compname)
                val = dict(zip((it.product(*[i.value for i in comp.domain])), val))
            elif not isinstance(val, dict):
                val = val.to_dict()
            self.setup[compname].update(val)
            if self._mode == MODE_SIM:
                SimRule(compname).eval(self)
        return self

    def new(self, comptype, *args, **kwargs):
        if kwargs.get('only', self._mode) != self._mode:
            logger.debug(f"Skipping '{kwargs.get('doc','Undocumented item')}', not supported in this mode")
            return None
        q = Queue()
        self._queue.append(q)
        neweqfrominit = self._mode == MODE_SIM
        neweqdeferred_default = False
        if comptype == CALIBPARAM:
            neweqfrominit = neweqfrominit or (self._calib)
            neweqdeferred_default = self._calib
            comptype = Model.calib2calibparam_comptype[self._calib]
        if comptype in [PARAM, VAR]:
            finit = kwargs.pop('initialize', None)
            if finit is not None:
                if neweqfrominit:
                    kwargs2 = dict(kwargs)
                    if 'eval_just_once' not in kwargs2:
                        kwargs2['eval_just_once'] = (not neweqdeferred_default)
                    q.put(lambda varname, varobj: (('{varname}{comptype}eq'
                                                    .format(varname=varname,comptype=comptype)),
                                                    self.new(EQUATION, varobj, finit, **kwargs2)))
                elif self._mode == MODE_OPT:
                    kwargs['initialize'] = wrap_pyomo_finit(finit)
            if self._mode == MODE_SIM:
                q.put(lambda varname, varobj: ('{varname}rule'.format(varname=varname), SimRule(varname)))
        orig_args = list(args)
        if (comptype in [PARAM,VAR,EQUATION,CONSTRAINT]):
            nsow = self._sow_setup.get(kwargs.get('doc', None), kwargs.get('sow', self._default_sow))
            parallel = kwargs.get('parallel', 0)
            orig_parallel = parallel
            if (nsow>0):
                if comptype in [PARAM,VAR]:
                    args += (self.sow,)
                parallel += 1
            kwargs['parallel'] = parallel
        if (comptype == VAR):
            if ('bounds' not in kwargs):
                kwargs['bounds'] = (None, None)
            if (self._mode == MODE_SIM) and (kwargs['bounds'] != (None, None)):
                kwargs3 = dict(kwargs)
                kwargs3['parallel'] = orig_parallel+1
                bnds = kwargs3.pop('bounds')
                if callable(bnds):
                    kwargs3['initialize'] = bnds
                    kwargs3.pop('default', None)
                else:
                    bnds = np.array(bnds)
                    assert bnds.shape == (2,)
                    for a in orig_args:
                        #prev_bnds = bnds
                        bnds = np.repeat(bnds[np.newaxis, :], len(a), axis=0)
                        #bnds[a,:] = prev_bnds
                    kwargs3['default'] = bnds  # np.newaxis #{i: bnds[i] for i in range(2)}
                    kwargs3.pop('initialize', None)
                kwargs3['sow'] = nsow
                q.put(lambda varname, varobj: ('{varname}_bounds'.format(varname=varname),
                                               self.new(PARAM, *orig_args, self._model._setlubounds,
                                                        **kwargs3)))
        else:
            kwargs.pop('bounds', None)
        for k in Model.mode2kwargs2pop[self._mode]:
            kwargs.pop(k, None)
        value = Model.mode2component[comptype][self._mode](*args, **kwargs)
        value._idnum = self._idnum
        value._comptype = comptype
        self._idnum += 1
        return value

    def add(self, comp, vals):
        if not isinstance(vals, list):
            vals = [vals]
        if not isinstance(comp, str):
            comp = comp.name
        prevset = self.setup.get(comp, {None: getattr(self, comp).initialize})
        newlist = prevset[None]
        newlist.extend(vals)
        self.set(comp, newlist)
        return self

    def __getitem__(self, key):
        return getattr(self._model.d, key)

    def _post_process(self):
        pass

    def solve(self, **kwargs):
        last_inst_solved = self._model.solve(**kwargs)
        ret = last_inst_solved.d
        ret.inst = last_inst_solved
        self._after_solve(ret)
        return ret

    def solve_ret_self(self, **kwargs):
        self.solve(**kwargs)
        return self

    def getvalue(self, v, t):
        if isinstance(v, float) or isinstance(v, int):
            return float(v)
        if isinstance(v, pd.Series):
            return v[t]
        return self._model.getvalue(v, t)

    def fix_viaparam(self, varname, value):
        # TODO
        self.set(varname, {i: v for i, v in value.iteritems()})
        for i, v in value.iteritems():
            self.setfixed(varname, i)
        return self

    def fix_viarule(self, varname, value, force=False):
        def newrule(m, t):
            v = getattr(m, varname)
            if m.isfixed(v, t) and (not force):
                return Expression.Skip
            m.setfixed(v, t)
            return m.getvalue(value, t)
        self.add_rule(varname, newrule)
        return self

    def fix(self, varname, value, force=True):
        if isinstance(value, str):
            value = Data.load(value)
        if isinstance(value, Data):
            value = getattr(value, varname)
        return self.fix_viarule(varname, value, force=force)

    def add_rule(self, varname, rule, force=True):
        self._model.add_rule(varname, rule, force)
        return self


class SimProblem(Problem):

    def __init__(self, m):
        super().__init__()


class MultiModel(object):

    def __init__(self, *args):
        for m in args:
            assert isinstance(m, Model) and (m._mode == MODE_SIM)
        self._mlist = args
        self._last_tracked_iterlist = []
        self._n = len(args)

    def run(self, *args, times=None, track=False, **kwargs):
        if times is None:
            times = [m.time for m in self._mlist]

        # Compute end of time horizon of interest
        #run_tend = min(t.end for t in times)

        # Initialize iterators
        i = [None]*self._n  # iterators
        i[0] = self._mlist[0].step(*args, time=times[0], **kwargs)
        for j in range(1, self._n):
            i[j] = self._mlist[j].step(time=times[j])

        # Simulation loop
        if len(self._last_tracked_iterlist) == 0:
            models_onhold = []
            io_items_available = []
            for mcurr, icurr in zip(self._mlist, i):
                models_onhold.append((mcurr, icurr, None))
            #counter = -1
            while len(models_onhold)>0:
                #logger.debug('models on hold:\n'+"\n".join([str(i) for i in models_onhold]))
                mcurr, icurr, io_item_to_waitfor = models_onhold.pop(0)
                if (io_item_to_waitfor is None) or (io_item_to_waitfor in io_items_available):
                    try:
                        if track:
                            self._last_tracked_iterlist.append(i.index(icurr))
                        outputs, input = next(icurr)
                        io_items_available.extend(outputs)
                        models_onhold.append((mcurr, icurr, input))
                    except StopIteration:
                        pass
                else:
                    models_onhold.append((mcurr, icurr, io_item_to_waitfor))
        else:
            for iicurr in self._last_tracked_iterlist:
                try:
                    next(i[iicurr])
                except StopIteration:
                    pass

        for m in self._mlist:
            m._after_solve(m.d)

        return self

        """
                #active.push((mcurr, icurr), -mcurr.time.start)
        while len(active)>0:
            mcurr, icurr = active.pop()
            io_items = next(icurr)
            counter = 0
            while(len(io_items)>0):
                t, v, isoutput = io_items[counter]
                if v is None:
                    io_items.pop(counter)
                    counter = min(counter, len(io_items)-1)
                    continue
                if isoutput
                #for t, v, isoutput in io_items:
                if v is not None:
                    if not isoutput:
                        try:
                            m2, i2 = buffer[(v, t)] # if popped, one item couldn't signal to 2 waiting models
                            io_items.extend(next(icurr))
                            active.push((mcurr, icurr), -t)
                        except:
                            if (v,t) in onhold:
                                onhold[(v, t)].add((mcurr, icurr))
                            else:
                                onhold[(v, t)] = set([(mcurr, icurr)])
                    else:
                        try:
                            for m2, i2 in onhold.pop((v, t)):
                                active.push((m2, i2), -t)
                        except:
                            buffer[(v, t)] = (mcurr, icurr)
                        active.push((mcurr, icurr), -t)
                else:
                    active.push((mcurr, icurr), -t)

        if len(buffer) > 0:
            logger.info(f'Variables left to be consumed: {buffer}')
        assert len(onhold) == 0, f'Models left waiting for inputs from others: {onhold}'
        assert t >= run_tend, f'Mismatch between expected ({run_tend}) and actual ({t}) end time'
        """

    def __call__(self, *args):
        #logger.debug(f'{self}, computing {args[0]}')
        ret = self.run(*args)
        #logger.debug(f'{self}, got {ret}')
        return ret

    def run_and_ret_objs_list(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return [float(o.value) for o in self._active_objs]

    def run_and_ret_objs_constrs_list(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return ([float(o.value) for o in self._active_objs],
                [float(c.value) for c in self._active_constrs])

    def run_and_ret_objs(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return {o.name: float(o.value) for o in self._active_objs}

    def run_and_ret_single(self, x):
        idx, args = x
        args = list(args)
        self.run(*args)
        return (idx, {o.name: float(o.value) for o in self._active_objs})

    def run_parallel(self, xlist, ncpus=None):
        pool = ProcessingPool(ncpus=ncpus).map
        result = pool(self.run_and_ret_single, xlist)
        return dict(result)

    def get_active_xxx(self, xreq, xlist):
        bfound = False
        for m in self._mlist:
            req = getattr(m, xreq)
            if req is not None:
                bfound = True
                logger.info('Using {xreq} from {classname}'.format(xreq=xreq, classname=m.__class__.__name__))
                break
        if not bfound:
            logger.warn(f'No model explicitly specified {xreq}')
            return []
        bfound = False
        active_xxx = []
        for oname in req:
            for o in getattr(m, xlist):
                if o.name == oname:
                    bfound = True
                    break
            assert bfound, '"{oname}" not found in {xlist}'.format(oname=oname,xlist=xlist)
            active_xxx.append(o)
        assert len(req) == len(active_xxx), ('{xlist} requested:{sreq}\n'
                                                  '{xlist} found:{xfound}'
                                                  .format(xlist=xlist, sreq=",".join(req),
                                                          xfound=",".join(active_xxx)))
        return active_xxx

    def asproblem(self):
        active_constrs = self.get_active_xxx('_con', '_constrs')
        nconstrs = len(active_constrs)
        if nconstrs == 0:
            f = self.run_and_ret_objs_list
        else:
            f = self.run_and_ret_objs_constrs_list
        active_objs = self.get_active_xxx('_vout', '_objs')
        nobjs = len(active_objs)
        bounds_list = self._mlist[0].get_bounds()
        nvars = len(bounds_list)
        p = Problem(nvars, nobjs, nconstrs=nconstrs, function=f)
        p.types[:] = [Real(bnds[0], bnds[1]) for bnds in bounds_list]
        p.constraints[:] = "<=0"
        p.directions[:] = [o.sense for o in active_objs]
        self._active_objs = active_objs
        self._active_constrs = active_constrs
        return p

    def asmodel(self):
        import rhodium.model as rh
        platypus2rhodium = {Problem.MINIMIZE: rh.Response.MINIMIZE,
                            Problem.MAXIMIZE: rh.Response.MAXIMIZE}
        model = rh.Model(self.run_and_ret_objs)
        model.parameters = [rh.Parameter(v) for v in self._vin]

        bfound = False
        for m in self._mlist:
            if m._vout is not None:
                bfound = True
                logger.info('Using objectives from {classname}'.format(classname=m.__class__.__name__))
                break
        assert bfound, 'No model explicitly specified objectives to optimize'
        active_objs = [o for o in m._objs if m.add_obj_if(o)]
        model.responses = [rh.Response(o.name, platypus2rhodium[o.sense]) for o in active_objs]

        # Define any constraints (can reference any parameter or response by name)
        # model.constraints = [Constraint("reliability >= 0.95")]

        # Some parameters are levers that we control via our policy
        levers = []
        for v in self._vin:
            bnds = self.get_bounds(v)
            levers.append(rh.RealLever(v, min(b[0] for b in bnds), max(b[1] for b in bnds), length=len(bnds)))
        model.levers = levers

        model.uncertainties = []
        self._active_objs = active_objs
        return model

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, item):
        for i, m in enumerate(self._mlist):
            try:
                x = getattr(m, item)
                try:
                    if x.isdynamic:
                        assert len(m.year) in x.shape
                        if ('_year' not in item):
                            item += '_year'
                    ret = m.d[item]
                except:
                    ret = x
            except:
                continue
            logger.info('"{item}" from {classname}'.format(item=item,classname=m.__class__.__name__))
            return ret
        raise KeyError('{item} not found in any model'.format(item=item))


class BiModel(object):

    def __init__(self, m1, m2):
        self._m1 = m1
        self._m2 = m2
        #for m in args:
        #    assert m._mode == MODE_SIM, 'Only MODE_SIM supported for the moment'

    def __getattr__(self, name):
        if (name[0] == '_'):
            return super().__getattr__(name)
        name += '_year'
        try:
            ret = getattr(self._m1._model.d, name)
            m = self._m1._model.d
        except:
            ret = getattr(self._m2._model.d, name)
            m = self._m2._model.d
        return ret

    def run(self, *args, kwargs1={}, kwargs2={}):
        i = [None, None]
        v = [None, None]
        t = [None, None]
        i[0] = self._m1.step(*args,**kwargs1)
        i[1] = self._m2.step(**kwargs2)
        t[0], v[0] = next(i[0])
        t[1], v[1] = next(i[1])
        try:
            while True:
                if (v[1] in self._m2._set_by_other_model):
                    jj = self._one2two(t[0], t[1])
                elif (v[1] in self._m2._given_to_other_model):
                    jj = self._two2one(t[0], t[1])
                else:
                    raise Exception('Models I/O not matching')
                for j in jj:
                    t[j], v[j] = next(i[j])
        except StopIteration:
            pass
        return [float(o.value) for o in self._m1._objs]

    def __call__(self, *args):
        #logger.debug(f'{self}, computing {args[0]}')
        ret = self.run(*args)
        #logger.debug(f'{self}, got {ret}')
        return ret

    def asproblem(self):
        p = self._m1.asproblem()
        p.function = self
        return p



