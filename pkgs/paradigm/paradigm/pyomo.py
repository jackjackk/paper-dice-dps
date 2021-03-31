from copy import deepcopy
import pprint
from pyomo.core.base import Objective
from pyomo.opt import SolverFactory, SolverManagerFactory
import pyomo.environ as pe
from pyomo.core.base.param import Param, SimpleParam, IndexedParam
from pyomo.core.base.var import Var, IndexedVar, SimpleVar
from pyomo.opt.parallel.manager import solve_all_instances
from functools import partial
from pandas import Series, DataFrame, PeriodIndex, concat, set_option
from numpy import array
from types import SimpleNamespace
from itertools import chain, product
import logging
import os
import dill as pickle
#from model import Dice
#import seaborn as sb
import numpy as np

logger = logging.getLogger('solve')
set_option('display.large_repr', 'truncate')
set_option('display.max_columns', 0)
#model.pprint()


def insts2dict(insts, comp2include=[Param, Var, Objective], simplify=False):
    list_output = chain(*[chain(*[i.component_objects(x, active=True) for x in comp2include]) for i in insts])
    #tidx = {i: list(i.year.values()) for i in insts}
    # Prepare N/A values for quantities not defined for some scenarios
    nas = SimpleNamespace();
    nas.value = float('Nan')
    natv = SimpleNamespace();
    natp = SimpleNamespace();
    data = {}
    if simplify:
        assert len(insts) == 1
    for v in list_output:
        if v.name in data:
            continue
        if isinstance(v, SimpleVar) or isinstance(v, SimpleParam):
            y = Series([getattr(i, v.name, nas).value for i in insts], index=[i.name for i in insts])
            # y = v.value
        elif isinstance(v, Objective):
            y = Series([getattr(i, v.name, nas).expr() for i in insts], index=[i.name for i in insts])
        elif isinstance(v, IndexedVar) or isinstance(v, IndexedParam):
            if isinstance(v, IndexedVar):
                getval = lambda x: x.value
                natv.values = lambda i: [na for t in v.index_set()]
                na = natv
            else:
                getval = lambda x: x
                natp.values = lambda i: [float('Nan') for t in v.index_set()]
                na = natp
            y = DataFrame(
                {i.name: Series([getval(x) for x in getattr(i, v.name, na).values()], index=v.index_set()) for i in insts if
                 hasattr(i, v.name)})
            #y = y.loc[:2100]
        if simplify:
            y = y[insts[0].name]
            if isinstance(y, Series):
                y = [np.nan,]+list(y.values)
            # y = Series(v.values(), index=idx)
        #y.__call__ = lambda x: x.plot()
        data[v.name] = y
    return data


def solmerge(slist):
    for i,s in enumerate(slist):
        if isinstance(s, str):
            slist[i] = DiceSolution.load(s)
    ret = {}
    keys = set(chain(*[s.data.keys() for s in slist]))
    for k in keys:
        for s in slist:
            try:
                if isinstance(s.data[k], Series):
                    axis=0
                else:
                    axis=1
                break
            except:
                pass
        ret[k] = concat([s.data.get(k, None) for s in slist],axis)
    return DiceSolution(**ret)



class DiceSolution(object):

    def load(filename):
        if filename[-4:] != '.dat':
            filename += '.dat'
        try:
            path = os.path.realpath(filename)
            with open(path, 'rb') as f:
                sloaded = pickle.load(f)
        except:
            try:
                path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', filename)
                with open(path, 'rb') as f:
                    sloaded = pickle.load(f)
            except:
                raise Exception('{filename} not found'.format(filename=filename))
        logger.info('Loaded "{path}"'.format(path=path))
        return sloaded


    def __init__(self, file=None, **kwargs):
        if file is not None:
            self.data = DiceSolution.load(file).data
            assert len(kwargs) == 0
        else:
            self.data = kwargs


    def save(self, name):
        path = '{name}.dat'.format(name=name)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info('Saved to "{path}"'.format(path=path))
        return self


class DiceResults(DiceSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            m = self.data
            years = m['year'].iloc[:,0].values
            sum_over_century = lambda x: x.set_index(years).reindex(range(years[0],years[-1]+1)).loc[:2100].interpolate().sum()
            m['PPMATM'] = m['MAT']/2.13
            m['EBASE'] = m['sigma']*m['YGROSS']
            m['TOT_EABAT'] = sum_over_century(m['EABAT'])
            vcosts = [v for v in m.keys() if v[-4:] == 'COST']
            for v in vcosts:
                m['{v}_PCTGDP'.format(v=v)] = m[v].div(m['YGROSS'])
            t = m['year']
            m['NPV_discount'] = (1./(1.+m['RI'])).pow(range(0, t.iloc[-1][0]-m['baseyear'][0]+1, m['tstep'][0]))
            m['NPV_C'] = m['C'].mul(m['NPV_discount']).sum()
            for scen in ['bau','opt']:
                try:
                    yref = DiceSolution.load(scen)
                    m['NPV_C_DIFF_{scen}'.format(scen=scen.upper())] = yref.data['NPV_C'] - m['NPV_C']
                    m['NPV_C_DIFFPCT_{scen}'.format(scen=scen.upper())] = (yref.data['NPV_C'] - m['NPV_C'])/(yref.data['NPV_C'])
                except:
                    m['NPV_C_DIFF_{scen}'.format(scen=scen.upper())] = 0 * m['C']
                    m['NPV_C_DIFFPCT_{scen}'.format(scen=scen.upper())] = 0 * m['C']
        except:
            logger.info('Skipping DICE post-processing')
            pass



class PyomoRun(DiceResults):
    def __init__(self, insts, res, levers, **kwargs):
        self.insts = insts
        self.res = res
        self.levers = levers
        try:
            self.report = (concat(
                [DataFrame({x: {'Problem.{y}'.format(y=y): v.value for y, v in res[x]['Problem'][0].items() if str(v.value) != '<undefined>'} for x in res.keys()}),
                 DataFrame({x: {'Solver.{y}'.format(y=y): getattr(v, 'value', '<undefined>') for y, v in res[x]['Solver'][0].items() if str(getattr(v, 'value', '<undefined>')) != '<undefined>' } for x in res.keys()})])
                .replace(r'3.12.8[^\s]+ (.+)$', r'\1', regex=True))
            # Print solution report
            logger.info(self.report)
        except:
            logger.info('Unable to save report')

        kwargs.update(insts2dict(insts))
        super().__init__(**kwargs)

    def save(self, name=None):
        if name is None:
            name = self.insts[0].name
        return super().save(name)

        """
        for v in list_output[1]:
            y = [x.value for x in v.values()]
            if v._bounds_init_rule is not None:
                m = inst.model()
                lo,up = list(map(list, zip(*[inst.MIU._bounds_init_rule(m, t) for t in inst.t])))
                if isinstance(up[0], Param):
                    up = [x.value for x in up]
                y = {'l':y, 'lo':lo, 'up':up}
                setattr(self, v.name, DataFrame(y, index=idx))
            else:
                setattr(self, v.name, Series(y, index=idx))
        """

class PyomoSolver(object):

    def __init__(self, solver='ipopt', max_iter=100000, max_cpu_time=600, linear_solver='mumps', logfile=None, tol=1e-9, **kwargs):
        #try:
        #    ipopt_path = os.path.join(os.environ['HOME'], 'tools', 'Ipopt-3.12.8', 'build')
        #    os.environ['LD_LIBRARY_PATH'] += ':'+os.path.join(ipopt_path, 'lib')
        #    kwargs['executable'] = os.path.join(ipopt_path, 'bin', 'ipopt')
        #except:
        #    pass
        opt = SolverFactory(solver, **kwargs)
        if opt is None:
            raise ValueError(f"Problem constructing solver {solver}")
        # Set options
        opt.options['max_iter'] = max_iter
        opt.options['max_cpu_time'] = max_cpu_time
        opt.options['linear_solver'] = linear_solver
        opt.options['tol'] = tol
        opt.options['halt_on_ampl_error'] = 'yes'
        # Save as attributes
        self.opt = opt
        self.solman = SolverManagerFactory('serial')

    def __call__(self, m, **kwargs):
        # kwargs = dict(tee=True, timelimit=None, logfile=None, ...)
        i = m.create_instance() #deepcopy(m).create_instance()
        #i.pprint()
        hand_queued = self.solman.queue(i, opt=self.opt, **kwargs)
        hand_returned = self.solman.wait_any()
        assert hand_queued == hand_returned
        results = self.solman.get_results(hand_returned)
        i.report = (concat(
                [Series({'Problem.{y}'.format(y=y): v.value for y, v in results['Problem'][0].items() if str(v.value) != '<undefined>'}),
                 (Series({'Solver.{y}'.format(y=y): getattr(v, 'value', '<undefined>') for y, v in results['Solver'][0].items() if str(getattr(v, 'value', '<undefined>')) != '<undefined>' })
                .replace(r'3.12.8[^\s]+ (.+)$', r'\1', regex=True))]))
        # Print solution report
        logfile = kwargs.get('logfile', None)
        if logfile in kwargs:
            with open(kwargs['logfile'], "a") as f:
                f.write(str(i.report))
        logger.debug('Solver report:\n{report}\n'.format(report=pprint.pformat(i.report)))
        return i


class Experiment(object):
    def __init__(self, problems, apply=[], solver='ipopt', max_iter=100000, linear_solver='mumps', logfile=None, tol=1e-9):
        super().__init__()
        if not isinstance(problems, list):
            problems = [problems]
        # Create solver
        kwds = {}
        #assert 'ipopt' in os.environ['LD_LIBRARY_PATH'].lower()
        #os.environ['LD_LIBRARY_PATH'] += ':/home/jack/tools/Ipopt-3.12.8/build/lib'
        #kwds['executable'] = os.path.join(os.environ['HOME'],'tools','Ipopt-3.12.8','build','bin','ipopt')
        #except:
        #    pass
        opt = SolverFactory(solver, **kwds)
        if opt is None:
            raise ValueError(f"Problem constructing solver {solver}")
        # Set options
        opt.options['max_iter'] = max_iter
        opt.options['linear_solver'] = linear_solver
        opt.options['tol'] = tol
        opt.options['halt_on_ampl_error'] = 'yes'
        #opt.options['print_level'] = 12
        #opt.options['print_info_string'] = 'yes'
        # Save as attributes
        self.opt = opt
        self.logfile = logfile
        self.problems = problems
        flist = apply
        if not isinstance(flist, list):
            flist = [flist]
        self.flist = flist


    def run(self, twopass=False, tee=True, parallel=False, levers=[[None]]):
        # send them to the solver(s)
        self.insts = []
        levers_expanded = []
        def class_set_scalar(m, p, pval):
            m.set_scalar(p=p, pval=pval)
            return m
        for lev in levers:
            if lev == [None]:
                levers_expanded.append((None,))
                break
            if isinstance(lev[1][0], int):
                rootname = lev[0]+'={}'
            elif isinstance(lev[1][0], float):
                rootname = lev[0] + '={:.2f}'
            else:
                raise Exception('Range values not recognized')
            #f = lev[0]
            farg = lev[0] #lev[2]
            rng = lev[1] #lev[3]
            levers_expanded.append([(rootname.format(x), partial(class_set_scalar, p=farg, pval=x)) for x in np.arange(rng[0], rng[1], rng[2])])
            #levers_expanded.append([(rootname.format(x), rootname, x) for x in rng])
        for p, lev in product(self.problems, product(*levers_expanded)):
            if lev != (None,):
                lev_flist = [l[1] for l in lev]
                lev_suffname = ';'+';'.join(['{l0}'.format(l0=l[0]) for l in lev])
            else:
                lev_flist = []
                lev_suffname = ''
            i = deepcopy(p).apply_list(self.flist+list(lev_flist)).create_instance()
            i.name += lev_suffname
            self.insts.append(i)
            logger.info('Created "{iname}"'.format(iname=i.name))

        # Create solver manager
        if not parallel:
            solman = SolverManagerFactory('serial')
            #solve_function = partial(solman.solve, insts[0],
            #                         opt=opt, tee=True, timelimit=None, logfile=logfile)
        else:
            solman = SolverManagerFactory('pyro')
            #solve_function = self.solve_parallel #partial(solve_all_instances, solman, opt, insts)
        self.solman = solman

        # Submit jobs
        self.handles = []
        self.handle2name = {}
        for i in self.insts:
            ahand = self.solman.queue(i, opt=self.opt, tee=tee, timelimit=None, logfile=self.logfile)
            self.handles.append(ahand)
            self.handle2name[ahand] = i.name

        results = {}
        # retrieve the solutions
        for i, inst in enumerate(self.insts):  # we know there are two instances
            this_action_handle = self.solman.wait_any()
            solved_name = self.handle2name[this_action_handle]
            results[solved_name] = self.solman.get_results(this_action_handle)
            #inst.postprocess()
            print("Solved", solved_name)
        #print(results)
        levers = [('name', [p.name for p in self.problems]),]+levers
        return PyomoRun(self.insts, results, levers)


    def solve2pass(self, model, logfile=None, tnum=20):
        inst = model.create_instance()
        prev_tnum = model.tnum.value
        prev_tstep = model.tstep.value
        prev_tot = prev_tnum*prev_tstep
        model.tnum.value = tnum
        assert prev_tot % tnum == 0
        model.tstep = prev_tot/tnum
        inst_aux = model.create_instance()
        logger.info('First pass')
        res1 = self.solman.solve(inst_aux, opt=self.opt, tee=True, timelimit=None, logfile=logfile)
        for v in inst_aux.component_objects(pe.Var, active=True):
            valprev = 0.
            atimeprev = 2010
            tt = 1
            for t, vx in v.iteritems():
                atime = 2010 + (t - 1) * (inst_aux.tstep.value)
                if vx.value is None:
                    continue
                valdelta = vx.value - valprev
                if not vx.fixed:
                    while (2010 + (tt - 1) * inst.tstep.value <= atime) and (tt <= inst.tnum.value):
                        btime = 2010 + (tt - 1) * inst.tstep.value
                        if atime > atimeprev:
                            getattr(inst, v.name)[tt] = valprev + valdelta * float(btime - atimeprev) / float(
                                atime - atimeprev)
                        else:
                            getattr(inst, v.name)[tt] = valdelta
                        tt += 1
                atimeprev = atime
                valprev = vx.value
        logger.info('Second pass')
        sol2 = self.solve(inst)
        return sol2


"""
res = solman.solve(inst, opt=opt, tee=True, timelimit=None, logfile='dice_full2.log')
inst.ipopt_zL_in.update(inst.ipopt_zL_out)
inst.ipopt_zU_in.update(inst.ipopt_zU_out)
opt.options['warm_start_init_point'] = 'yes'
opt.options['warm_start_bound_push'] = 1e-6
opt.options['warm_start_mult_bound_push'] = 1e-6
opt.options['mu_init'] = 1e-6
opt.options['nlp_scaling_method'] = 'none'
inst.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)
inst.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
res = solman.solve(inst, opt=opt, tee=True, timelimit=None, logfile='dice_full3.log')

pd.Series(inst.EIND.get_values()).plot()

#for v in inst.component_objects(pe.Var, active=True):
#    print v.name, pd.Series(v.get_values())[100]
#print ', '.join([v.name for v in inst.component_objects(pe.Var, active=True)])

# Create simplified instance


# Solve model
pd.Series(inst.EIND.get_values()).plot()
res.write()
inst.EIND.get_values()
    b = pd.Series(a.values, [2010+i*inst2.tstep.value for i in range(inst.tnum.value)])

        if not vx.fixed:

        print vx.value
        break
    break

a=list(inst.component_objects(pe.Var, active=True))[0]

inst.S
pd.Series([inst.S[i].value for i in inst.t], index=[2010+i*inst.tstep.value for i in range(inst.tnum.value)])
res.write()
"""

