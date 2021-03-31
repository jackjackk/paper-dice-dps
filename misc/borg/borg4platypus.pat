From c7cda432538f9c0e2b1ca403e52bf4c93da64215 Mon Sep 17 00:00:00 2001
From: jackjackk <jackjackk@gmail.com>
Date: Fri, 8 Feb 2019 14:29:07 +0100
Subject: [PATCH] add support for borg4platypus

---
 plugins/Python/borg.py | 198 ++++++++++++++++++++++-------------------
 1 file changed, 107 insertions(+), 91 deletions(-)

diff --git a/plugins/Python/borg.py b/plugins/Python/borg.py
index 181a2e2..a4c9e99 100644
--- a/plugins/Python/borg.py
+++ b/plugins/Python/borg.py
@@ -28,7 +28,8 @@ program.
     Evolutionary Computing Framework."  Evolutionary Computation,
     21(2):231-259.
 
-Copyright 2013-2018 David Hadka
+Copyright 2013-2014 David Hadka
+2019/01: Changes by Giacomo Marangoni to support borg4platypus
 Requires Python 2.5 or later
 """
 
@@ -36,13 +37,20 @@ from ctypes import *
 import os
 import sys
 import time
+from io import UnsupportedOperation
+from ctypes.util import find_library as _find_library
+
+
+def find_library(s):
+    for p in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
+        p = os.path.join(p,'lib{s}.so'.format(s=s))
+        if os.path.exists(p):
+            return p
+    return _find_library(s)
+
 
 terminate=False
 
-# For Python 3 compatibility.  In Python 3, it appears we must also explicitly set the restype for pointers
-# as c_void_p and also wrap the returned value in c_void_p(...).
-if sys.version_info > (3,):
-    long = int
 
 class Configuration:
     """ Holds configuration options for the Borg MOEA Python wrapper. """
@@ -79,25 +87,24 @@ class Configuration:
         using the Borg MOEA.
         """
 
+
         if path:
             Configuration.libc = CDLL(path)
-        elif os.name == "posix":
-            try:
-                Configuration.libc = CDLL("libc.so.6")
-            except OSError:
-                return
         elif os.name == "nt" and cdll.msvcrt:
             Configuration.libc = cdll.msvcrt
         else:
-            return
+            Configuration.libc = CDLL(find_library('c'))
 
         try:
             Configuration.stdout = Configuration.libc.fdopen(sys.stdout.fileno(), "w")
         except AttributeError:
             Configuration.stdout = Configuration.libc._fdopen(sys.stdout.fileno(), "w")
+        except UnsupportedOperation:
+            Configuration.stdout = sys.stdout
+
 
     @staticmethod
-    def setBorgLibrary(path=None):
+    def setBorgLibrary(path='borg'):
         """ Override the location of the Borg MOEA shared object.
 
         If the path is not specified, this method attempts to auto-detect the location
@@ -106,43 +113,23 @@ class Configuration:
         user to manually invoke this method before using the Borg MOEA
         """
 
-        if path:
-            try:
-                Configuration.libborg = CDLL(path)
-                Configuration.libborg.BORG_Copyright
-                Configuration.stdcall = False
-            except AttributeError:
-                # Not using __cdecl, try __stdcall instead
-                if os.name == "nt":
-                    Configuration.libborg = WinDLL(path)
-                    Configuration.stdcall = True
-        elif os.name == "posix":
-            try:
-                Configuration.libborg = CDLL("./libborg.so")
-                Configuration.stdcall = False
-            except OSError:
-                return
-        elif os.name == "nt":
-            try:
-                Configuration.libborg = CDLL("./borg.dll")
-                Configuration.libborg.BORG_Copyright
-                Configuration.stdcall = False
-            except OSError:
-                return
-            except AttributeError:
-                # Not using __cdecl, try __stdcall instead
-                try:
-                    Configuration.libborg = WinDLL("./borg.dll")
-                    Configuration.stdcall = True
-                except OSError:
-                    return
-                    
+        path = find_library(path)
+        if path is None:
+            raise OSError('Unable to locate "{}" lib'.format(path))
+
+        try:
+            Configuration.libborg = CDLL(path)
+            Configuration.libborg.BORG_Copyright
+            Configuration.stdcall = False
+        except AttributeError:
+            # Not using __cdecl, try __stdcall instead
+            if os.name == "nt":
+                Configuration.libborg = WinDLL(path)
+                Configuration.stdcall = True
+            else:
+                raise Exception('Unable to load borg lib')
+
         # Set result type of functions with non-standard types
-        Configuration.libborg.BORG_Problem_create.restype = c_void_p
-        Configuration.libborg.BORG_Operator_create.restype = c_void_p
-        Configuration.libborg.BORG_Algorithm_create.restype = c_void_p
-        Configuration.libborg.BORG_Algorithm_get_result.restype = c_void_p
-        Configuration.libborg.BORG_Archive_get.restype = c_void_p
         Configuration.libborg.BORG_Solution_get_variable.restype = c_double
         Configuration.libborg.BORG_Solution_get_objective.restype = c_double
         Configuration.libborg.BORG_Solution_get_constraint.restype = c_double
@@ -156,7 +143,7 @@ class Configuration:
         if value:
             Configuration.libborg.BORG_Random_seed(c_ulong(value))
         else:
-            Configuration.libborg.BORG_Random_seed(c_ulong(os.getpid()*long(time.time())))
+            Configuration.libborg.BORG_Random_seed(c_ulong(os.getpid()*int(time.time())))
 
     @staticmethod
     def enableDebugging():
@@ -190,27 +177,31 @@ class Configuration:
         except AttributeError:
             # The serial Borg MOEA C library is loaded; switch to parallel
             try:
-                Configuration.setBorgLibrary("./libborgmm.so")
+                Configuration.setBorgLibrary("borgmm")
             except OSError:
                 try:
-                    Configuration.setBorgLibrary("./libborgms.so")
+                    Configuration.setBorgLibrary("borgms")
                 except OSError:
                     raise OSError("Unable to locate the parallel Borg MOEA C library")
 
         # The following line is needed to load the MPI library correctly
-        CDLL("libmpi.so.0", RTLD_GLOBAL)
+        CDLL(find_library('mpi'), RTLD_GLOBAL)
 
         # Pass the command-line arguments to MPI_Init
-        argc = c_int(len(sys.argv))
-        CHARPP = c_char_p * len(sys.argv)
-        argv = CHARPP()
+        # see https://mail.python.org/pipermail/python-list/2016-June/709889.html
+        CHARP = POINTER(c_char)
+        CHARPP = POINTER(CHARP)
 
-        for i in range(len(sys.argv)):
-            argv[i] = sys.argv[i]
+        Configuration.argc = c_int(len(sys.argv))
+        Configuration.argv = (CHARP * (len(sys.argv) + 1))()
+        for i, arg in enumerate(sys.argv):
+            enc_arg = arg.encode()
+            Configuration.argv[i] = create_string_buffer(enc_arg)
 
         Configuration.libborg.BORG_Algorithm_ms_startup(
-            cast(addressof(argc), POINTER(c_int)),
-            cast(addressof(argv), POINTER(CHARPP)))
+            cast(addressof(Configuration.argc), POINTER(c_int)),
+            cast(addressof(Configuration.argv), POINTER(CHARPP))
+        )
 
         Configuration.startedMPI = True
 
@@ -312,12 +303,13 @@ class Borg:
         self.function = _functionWrapper(function, numberOfVariables, numberOfObjectives, numberOfConstraints, directions)
 
         if Configuration.stdcall:
-            self.CMPFUNC = WINFUNCTYPE(c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double))
+            self.CMPFUNC = WINFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double))
         else:
-            self.CMPFUNC = CFUNCTYPE(c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double))
+            self.CMPFUNC = CFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double))
 
         self.callback = self.CMPFUNC(self.function)
-        self.reference = c_void_p(Configuration.libborg.BORG_Problem_create(c_int(numberOfVariables), c_int(numberOfObjectives), c_int(numberOfConstraints), self.callback))
+        Configuration.libborg.BORG_Problem_create.restype = c_void_p
+        self.reference = c_void_p(Configuration.libborg.BORG_Problem_create(numberOfVariables, numberOfObjectives, numberOfConstraints, self.callback))
 
         if bounds:
             self.setBounds(*bounds)
@@ -383,7 +375,7 @@ class Borg:
         Configuration.libborg.BORG_Problem_set_bounds(self.reference, index, c_double(lowerBound), c_double(upperBound))
 
     def solveMPI(self, islands=1, maxTime=None, maxEvaluations=None, initialization=None, runtime=None,
-            allEvaluations=None):
+            allEvaluations=None, frequency=None):
         """ Runs the master-slave or multi-master Borg MOEA using MPI.
 
         islands        - The number of islands
@@ -395,6 +387,7 @@ class Borg:
         allEvaluations - Filename pattern for saving all evaluations (the filename should include
                          one %d which gets replaced by the island index).  Since this can quickly
                          generate large files, use this option with caution.
+        frequency      - The number of function evaluations between runtime outputs
         
         Note: All nodes must invoke solveMPI.  However, only one node will return the discovered
         Pareto optimal solutions.  The rest will return None.
@@ -422,18 +415,48 @@ class Borg:
             Configuration.libborg.BORG_Algorithm_ms_max_evaluations(c_int(maxEvaluations))
 
         if initialization and islands > 1:
-            Configuration.libborg.BORG_Algorithm_ms_initialization(c_int(initialization));
+            Configuration.libborg.BORG_Algorithm_ms_initialization(c_int(initialization))
 
         if runtime:
-            Configuration.libborg.BORG_Algorithm_output_runtime(c_char_p(runtime));
+            Configuration.runtime = runtime.encode()
+            Configuration.libborg.BORG_Algorithm_output_runtime(c_char_p(Configuration.runtime));
+
+        if frequency:
+            Configuration.libborg.BORG_Algorithm_output_frequency(c_int(frequency));
 
         if allEvaluations:
             Configuration.libborg.BORG_Algorithm_output_evaluations(c_char_p(allEvaluations));
 
-        result = Configuration.libborg.BORG_Algorithm_ms_run(self.reference)
+        Configuration.libborg.BORG_Algorithm_ms_run.restype = c_void_p
+        result = c_void_p(Configuration.libborg.BORG_Algorithm_ms_run(self.reference))
 
         return Result(result, self) if result else None
 
+
+    def write_runtime_header(self, settings):
+        runtimeformat = settings.get('runtimeformat', 'optimizedv')
+        fp = open(settings['runtimefile'], 'w')
+        if runtimeformat == 'optimizedv':
+            fp.write("//")
+            dynamics_header = [
+                "NFE", "ElapsedTime",
+                "SBX", "DE", "PCX", "SPX", "UNDX", "UM",
+                "Improvements", "Restarts",
+                "PopulationSize", "ArchiveSize"]
+            if settings.get("restartMode", None) == RestartMode.ADAPTIVE:
+                dynamics_header.append("MutationIndex")
+            fp.write(",".join(dynamics_header))
+            fp.write("\n")
+            header = ["NFE"] \
+                     + ["dv{0}".format(i) for i in range(self.numberOfVariables)] \
+                     + ["obj{0}".format(i) for i in range(self.numberOfObjectives)] \
+                     + ["con{0}".format(i) for i in range(self.numberOfConstraints)]
+            fp.write(",".join(header))
+            fp.write("\n")
+            fp.flush()
+        return fp, dynamics_header
+
+
     def solve(self, settings={}):
         """ Runs the Borg MOEA to solve the defined optimization problem, returning the
         discovered Pareto optimal set.
@@ -448,7 +471,7 @@ class Borg:
 
         maxEvaluations = settings.get("maxEvaluations", 10000)
         start = time.clock()
-
+        Configuration.libborg.BORG_Operator_create.restype = c_void_p
         pm = c_void_p(Configuration.libborg.BORG_Operator_create("PM", 1, 1, 2, Configuration.libborg.BORG_Operator_PM))
         Configuration.libborg.BORG_Operator_set_parameter(pm, 0, c_double(settings.get("pm.rate", 1.0 / self.numberOfVariables)))
         Configuration.libborg.BORG_Operator_set_parameter(pm, 1, c_double(settings.get("pm.distributionIndex", 20.0)))
@@ -477,6 +500,7 @@ class Borg:
         Configuration.libborg.BORG_Operator_set_parameter(undx, 0, c_double(settings.get("undx.zeta", 0.5)))
         Configuration.libborg.BORG_Operator_set_parameter(undx, 1, c_double(settings.get("undx.eta", 0.35)))
 
+        Configuration.libborg.BORG_Algorithm_create.restype = c_void_p
         algorithm = c_void_p(Configuration.libborg.BORG_Algorithm_create(self.reference, 6))
         Configuration.libborg.BORG_Algorithm_set_operator(algorithm, 0, sbx)
         Configuration.libborg.BORG_Algorithm_set_operator(algorithm, 1, de)
@@ -501,25 +525,7 @@ class Borg:
             lastSnapshot = 0
             frequency = settings.get("frequency")    
             if "runtimefile" in settings:
-                fp = open(settings['runtimefile'], 'w')
-                if runtimeformat == 'optimizedv':
-                    fp.write("//")
-                    dynamics_header = [
-                            "NFE", "ElapsedTime", 
-                            "SBX", "DE", "PCX", "SPX", "UNDX", "UM",
-                            "Improvements", "Restarts", 
-                            "PopulationSize", "ArchiveSize"]
-                    if settings.get("restartMode", None) == RestartMode.ADAPTIVE:
-                        dynamics_header.append("MutationIndex")
-                    fp.write(",".join(dynamics_header))
-                    fp.write("\n")
-                    header = ["NFE"] \
-                           + ["dv{0}".format(i) for i in range(self.numberOfVariables)] \
-                           + ["obj{0}".format(i) for i in range(self.numberOfObjectives)] \
-                           + ["con{0}".format(i) for i in range(self.numberOfConstraints)]
-                    fp.write(",".join(header))
-                    fp.write("\n")
-                    fp.flush()
+                fp, dynamics_header = self.write_runtime_header(settings)
             else:
                 fp = None
         else:
@@ -552,6 +558,10 @@ class Borg:
                 if fp is None:
                     statistics.append(entry)
                 else:
+                    if settings.get('runtimereset', True):
+                        fp.close()
+                        fp, dynamics_header = self.write_runtime_header(settings)
+                    Configuration.libborg.BORG_Algorithm_get_result.restype = c_void_p
                     archive = Result(c_void_p(Configuration.libborg.BORG_Algorithm_get_result(algorithm)), self, statistics)
                     if runtimeformat == 'optimizedv':
                         row = ["{0}".format(entry[dynamic]) for dynamic in dynamics_header]
@@ -596,9 +606,9 @@ class Borg:
 
                 lastSnapshot = currentEvaluations
 
+        Configuration.libborg.BORG_Algorithm_get_result.restype = c_void_p
         result = c_void_p(Configuration.libborg.BORG_Algorithm_get_result(algorithm))
-
-        if fp is not None:
+        if "runtimefile" in settings:
             fp.close()
 
         Configuration.libborg.BORG_Operator_destroy(sbx)
@@ -654,7 +664,7 @@ class Solution:
 
     def display(self, out=sys.stdout, separator=" "):
         """ Prints the decision variables, objectives, and constraints to standard output. """
-        print >> out, separator.join(map(str, self.getVariables() + self.getObjectives() + self.getConstraints()))
+        print(separator.join(map(str, self.getVariables() + self.getObjectives() + self.getConstraints())), file=out, end='\n')
 
     def violatesConstraints(self):
         """ Returns True if this solution violates one or more constraints; False otherwise. """
@@ -688,6 +698,7 @@ class Result:
 
     def get(self, index):
         """ Returns the Pareto optimal solution at the given index. """
+        Configuration.libborg.BORG_Archive_get.restype = c_void_p
         return Solution(c_void_p(Configuration.libborg.BORG_Archive_get(self.reference, index)), self.problem)
 
 class ResultIterator:
@@ -698,7 +709,10 @@ class ResultIterator:
         self.result = result
         self.index = -1
 
-    def next(self):
+    def __iter__(self):
+        return self
+
+    def __next__(self):
         """ Returns the next Pareto optimal solution in the set. """
         self.index = self.index + 1
 
@@ -707,7 +721,9 @@ class ResultIterator:
         else:
             return self.result.get(self.index)
 
-    __next__ = next
+    def next(self):
+        return self.__next__()
+    
 
 def _functionWrapper(function, numberOfVariables, numberOfObjectives, numberOfConstraints, directions=None):
     """ Wraps a Python evaluation function and converts it to the function signature
-- 
2.20.1

