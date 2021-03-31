import importlib
import copy
from pprint import pprint, pformat
from tempfile import TemporaryDirectory
from subprocess import Popen
from collections import namedtuple

import dill
from platypus.algorithms import Algorithm
from platypus.core import Solution, Problem
from tqdm import tqdm
import logging
import os
import sys
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("borg4platypus")
DEFAULT_EXTRA_BORGLIBDIR = ""


from platypus import Solution
import numpy as np


def _attrsolution(problem, attr):
    solution = Solution(problem)
    solution.variables[:] = [getattr(x, attr) for x in problem.types]
    solution.evaluate()
    return np.array(solution.objectives)


def _minsolution(problem):
    return _attrsolution(problem, 'min_value')


def _maxsolution(problem):
    return _attrsolution(problem, 'max_value')


def _load_default_libpath():
    for libpath in ['{prefix_libpath}_LIBRARY_PATH'.format(prefix_libpath=prefix_libpath) for prefix_libpath in ['DYLD', 'LD']]:
        os.environ[libpath] = os.environ.get(libpath, '') + os.pathsep + DEFAULT_EXTRA_BORGLIBDIR
_load_default_libpath()

from borgpy import Direction, InitializationMode


class BorgC(Algorithm):

    @staticmethod
    def load_default_libpath():
        _load_default_libpath()

    def __init__(self, problem, epsilons, seed=None, do_postprocess=True, do_solve=True, runtimedir=None,
                 log_frequency=None, name=None, liveplot=False, pbar=True, backend='borgpy.borg'):
        """
        Wraps the Python wrapper of BORG to make it compatible w/ Platypus.
        :param problem: Problem instance
        :param epsilons: objective space resolution
        :param log_frequency: frequency of output
        :param name: prefix string used for output filenames
        """
        super(BorgC, self).__init__(problem, evaluator=None, log_frequency=log_frequency, seed=seed)
        self.backend = backend
        self.settings = {}
        self.last_settings_passed_to_borg = {}
        if log_frequency is not None: self.settings['frequency'] = log_frequency
        self.cid = None
        if name is not None:
            if seed is not None:
                runtimefile = '{name}_seed{seed:04d}_runtime.csv'.format(name=name,seed=seed)
                counter = seed
            else:
                counter = 0
                while os.path.exists('{name}_run{counter:04d}_runtime.csv'.format(name=name,counter=counter)):
                    counter += 1
                runtimefile = '{name}_{counter:04d}_runtime.csv'.format(name=name,counter=counter)
                open(runtimefile, 'a').close()
            self.cid = counter
            self.settings['runtimefile'] = runtimefile
            #os.remove(self.settings["runtimefile"])
            #LOGGER.info(f'Removed {self.settings["runtimefile"]}')
            #self.name = f'{self.__class__.__name__}_{name}'
        if runtimedir is None:
            runtimedir = os.path.abspath(os.path.dirname(self.settings['runtimefile']))
        self.runtimedir = runtimedir
        os.chdir(runtimedir)
        LOGGER.info('runtimedir = {runtimedir}'.format(runtimedir=self.runtimedir))
        #problem = copy.deepcopy(problem)
        self.liveplot = liveplot
        self.pbar = pbar
        self.name = name
        self.seed = seed
        self.nfe = 0
        self.do_postprocess = do_postprocess
        self.do_solve = do_solve
        self.result = None
        self.borg = None
        #LOGGER.info(pformat(vars(self)))

        def problem_function(*vars):
            solution = Solution(problem)
            solution.variables[:] = vars
            solution.evaluate()
            constrs = [f(x) for (f, x) in zip(solution.problem.constraints, solution.constraints)]
            if pbar:
                try:
                    #print('problem',hex(id(problem)))
                    #print('borg_function',hex(id(problem.borg_function)))
                    problem.borg_function._progress_bar.update()
                except Exception as e:
                    #pass
                    #LOGGER.error(e, exc_info=True)
                    from mpi4py import rc
                    rc.initialize = False
                    from mpi4py import MPI
                    _cid = MPI.COMM_WORLD.Get_rank()
                    problem.borg_function._progress_bar = \
                        tqdm(range(problem.borg_function._max_evaluations),
                             desc='{_cid}'.format(_cid=_cid),
                             total=problem.borg_function._np, position=_cid)
                    problem.borg_function._progress_bar.update()
            return solution.objectives._data, constrs

        problem.borg_function = problem_function
        self.problem = problem
        platypus2borg_directions = {
            Problem.MAXIMIZE: Direction.MAXIMIZE,
            Problem.MINIMIZE: Direction.MINIMIZE
        }
        self.directions = [platypus2borg_directions[d] for d in problem.directions]

        '''
        solution_fields = ['variables', 'objectives']
        def problem_function_namedtuple(*vars):
            solution = namedtuple('Solution', solution_fields)
            solution.variables = vars
            solution.objectives = [0]*problem.nobjs
            problem.evaluate(solution)
            return solution.objectives
        '''

        if not isinstance(epsilons, list):
            epsilons = [epsilons]*problem.nobjs
        self.epsilons = epsilons

    def borg_init(self):
        LOGGER.info(f'Loading {self.backend} module')
        self.borg = importlib.import_module(self.backend)
        if self.seed is not None:
            self.borg.Configuration.seed(self.seed)
        borg_obj = self.borg.Borg(self.problem.nvars, self.problem.nobjs, self.problem.nconstrs, self.problem.borg_function, directions=self.directions)
        borg_obj.setBounds(*[[vtype.min_value, vtype.max_value] for vtype in self.problem.types])
        borg_obj.setEpsilons(*self.epsilons)
        return borg_obj

    def borg_deinit(self):
        self.borg = None

    def borg_solve(self, borg_obj):
        raise NotImplementedError('Please use one of parent classes, either SerialBorgC or MpiBorgC')

    def borg_result(self, borg_res):
        result = None
        if borg_res:
            result = []
            for borg_sol in borg_res:
                s = Solution(self.problem)
                s.variables[:] = borg_sol.getVariables()
                s.evaluate()
                try:
                    np.testing.assert_array_almost_equal(s.objectives[:], borg_sol.getObjectives())
                except Exception as e:
                    LOGGER.warning('x = {svariables}, {e}'.format(svariables=s.variables[:],e=e))
                result.append(s)
        return result

    def step(self):
        # Initialize Borg object
        borg_obj = self.borg_init()
        # Solve
        if self.do_solve:
            borg_res = self.borg_solve(borg_obj)
            # Process results (if any)
            if self.do_postprocess:
                self.result = self.borg_result(borg_res)
        # Run deinit routines (e.g. StopMPI) & unload module
        self.borg_deinit()
        # Let Platypus know we're done
        self.nfe = self.settings['maxEvaluations']

    def run(self, condition):
        assert isinstance(condition, int)
        self.settings['maxEvaluations'] = condition
        if self.liveplot:
            with open(os.path.join(self.runtimedir, 'liveplot_problem.dat'), 'wb') as f:
                dill.dump(self.problem, f)
            omin = _minsolution(self.problem)
            omax = _maxsolution(self.problem)
            pd.DataFrame({'omin':omin,'omax':omax}).T.to_csv(os.path.join(self.runtimedir, 'liveplot_reference.dat'))
            Popen([sys.executable, os.path.join(os.path.dirname(__file__), "liveplot.py"), self.runtimedir, '1', str(condition)])
        super(BorgC, self).run(condition)


class SerialBorgC(BorgC):
    def borg_solve(self, borg_obj):
        # Create progress bar
        if self.pbar:
            self.problem.borg_function._progress_bar = tqdm(range(self.settings['maxEvaluations']), position=self.cid)
        # Run BORG
        self.last_settings_passed_to_borg = self.settings.copy()
        return borg_obj.solve(self.last_settings_passed_to_borg)


class MpiBorgC(BorgC):
    def __init__(self, problem, epsilons, nproc=2, islands=1, **kwargs):
        super(MpiBorgC, self).__init__(problem, epsilons, **kwargs)
        self.settings['nProcs'] = nproc
        self.settings['islands'] = islands

    def borg_init(self):
        ret = super(MpiBorgC, self).borg_init()
        self.borg.Configuration.startMPI()
        return ret

    def borg_deinit(self):
        self.borg.Configuration.stopMPI()
        super(MpiBorgC, self).borg_deinit()

    def borg_solve(self, borg_obj):
        # Create progress bar
        self.problem.borg_function._max_evaluations = self.settings['maxEvaluations']
        self.problem.borg_function._np = self.settings['nProcs']
        #self.problem.borg_function._progress_bar = tqdm(range(self.settings['maxEvaluations']), position=self.cid)
        # Run BORG
        self.last_settings_passed_to_borg = self.settings.copy()
        # rename runtimefile -> runtime
        if self.name is not None:
            self.last_settings_passed_to_borg['runtime'] = os.path.basename(self.settings["runtimefile"])
        for k in ['runtimefile','nProcs']:
            try:
                self.last_settings_passed_to_borg.pop(k)
            except:
                pass
        self.last_settings_passed_to_borg['initialization'] = InitializationMode.GLOBAL_LATIN
        borg_res = borg_obj.solveMPI(**self.last_settings_passed_to_borg)
        return borg_res


class ExternalBorgC(BorgC):
    def __init__(self, problem, epsilons, np=None, tempdir=None, islands=1, **kwargs):
        """
        Instatiate Borg as a separate process.

        :param problem: [platypus.Problem] instance of problem to solve
        :param epsilons: [float|list[float]] objectives space resolution
        :param np: [None, int] number of Borg instances to launch for MPI, = $PBS_NP / 1 if None
        :param tempdir: [None, str] path to store input/output/temporary Borg files, made up if None
        :param kwargs: [dict] arguments to pass to inner Borg class
        """
        # Initialize following fields:
        # settings['frequency'], settings['runtimefile'], cid, runtimedir, liveplot, pbar,
        # name, seed, nfe, result, borg
        args = [problem, epsilons]
        super(ExternalBorgC, self).__init__(*args, **kwargs)
        # Save original args
        kwargs.pop('name', None)
        self.borg_kwargs = kwargs
        self.borg_args = args
        # Init MPI variables
        if np is None:
            np = int(os.environ.get('PBS_NP', '1'))
        LOGGER.debug('Number of cores: {np}'.format(np=np))
        self.np = np
        self.islands = islands
        # Init temp dir & file names
        if tempdir is None:
            tempdir = TemporaryDirectory()
        else:
            tempdir = namedtuple('GivenTemporaryDirectory', ['name'])(name=tempdir)
            if not os.path.exists(tempdir.name):
                os.mkdir(tempdir.name)
        self.tempdir = tempdir
        self.borg_runscript = os.path.join(self.tempdir.name, 'runscript.py')
        self.execlist = [self.borg_runscript]
        self.loglevel = LOGGER.getEffectiveLevel()



    def borg_init(self):
        # Serialize algorithm to run
        import dill
        self.pickle = dill
        if 'do_solve' in self.borg_kwargs:
            self.borg_kwargs['do_solve'] = True
        if self.np > 1:
            borg_class = MpiBorgC
            self.borg_kwargs['nproc'] = self.np
            self.borg_kwargs['islands'] = self.islands
            self.borg_launcher = ['mpirun', '-np', str(self.np)]
            if 'PBS_NODEFILE' in os.environ:
                self.borg_launcher.extend(['-machinefile', os.environ['PBS_NODEFILE']]) #+ self.mpirun.split(' ')
        else:
            borg_class = SerialBorgC
            self.borg_launcher = []
        if self.name is None:
            passed_name = None
        else:
            passed_name = os.path.join(os.getcwd(), self.name)
        algo2dump = borg_class(*self.borg_args,
                               name=passed_name, **self.borg_kwargs)
        self.borg_algo_dumpfile = os.path.join(self.tempdir.name, 'algo.dmp')
        self.borg_result_dumpfile = os.path.join(self.tempdir.name, 'result.dmp')
        with open(self.borg_algo_dumpfile, 'wb') as f:
            self.pickle.dump(algo2dump, f)
        #LOGGER.info(f'Dumped {algo2dump.__class__.__name__} object to "{self.borg_algo_dumpfile}"')
        with open(self.borg_runscript, 'w') as f:
            f.write("""
import dill
import sys
import logging
import os
logging.basicConfig(level={loglevel})
sys.path.append("{cwd}")
os.environ['LD_LIBRARY_PATH'] = "{ldlibpath}"
os.environ['DYLD_LIBRARY_PATH'] = "{dyldlibpath}"
with open("{borg_algo_dumpfile}", "rb") as f:
    algo = dill.load(f)
algo.mastertempdir = "{tempdirname}"
algo.run({maxeval})
if algo.result:
    with open("{borg_result_dumpfile}", "wb") as f:
        dill.dump(algo.result, f)                       
""".format(loglevel=self.loglevel, cwd=os.getcwd(), ldlibpath=os.environ['LD_LIBRARY_PATH'],
            dyldlibpath=os.environ['DYLD_LIBRARY_PATH'], borg_algo_dumpfile=self.borg_algo_dumpfile,
            tempdirname=self.tempdir.name, maxeval=self.settings['maxEvaluations'],
            borg_result_dumpfile=self.borg_result_dumpfile))
        return self.borg_runscript

    def borg_solve(self, borg_obj):
        import subprocess
        cmd_args = self.borg_launcher + [sys.executable,] + self.execlist
        cmd_args_string = ' '.join(cmd_args)
        LOGGER.info('Launching "{cmd_args_string}"'.format(cmd_args_string=cmd_args_string))
        proc = subprocess.Popen(cmd_args)
        stdout, stderr = proc.communicate()
        if self.borg_kwargs.get('process_result', True):
            with open(self.borg_result_dumpfile, 'rb') as f:
                res = self.pickle.load(f)
        else:
            res = None
        logdisplay_endsection = '-'*len('INFO:Platypus:Borg output-----')
        if stdout is not None:
            LOGGER.info('Borg output-----\n{stdout}\n{logdisplay_endsection}'.format(stdout=stdout,logdisplay_endsection=logdisplay_endsection))
        if stderr is not None:
            LOGGER.info('\n-----Borg error------\n{stderr}\n{logdisplay_endsection}'.format(stderr=stderr,logdisplay_endsection=logdisplay_endsection))
        return res

    def borg_result(self, borg_res):
        return borg_res

    def borg_deinit(self):
        if isinstance(self.tempdir, TemporaryDirectory):
            self.tempdir.cleanup()
        self.tempdir = None
        pass
