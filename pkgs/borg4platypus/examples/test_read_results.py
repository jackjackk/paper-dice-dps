import dill
import os

from paradice.dice import Dice, ECS

#os.chdir('examples')

with open('examples/dice/results_dps5_nfe100_seeds1_objs2.dat', 'rb') as f:
    dc, mdice, results = dill.load(f)

with open('examples/dice/results_temporal_nfe1000_seeds1_objs2.dat', 'rb') as f:
    dct, mdicet, resultst = dill.load(f)

dct.run(resultst[0][0]['MIU'])
mdice.responses.keys()
import rhodium as rh
rh.scatter2d(mdice, results[0], colors='r')
rh.scatter2d(mdicet, resultst[0], colors='b')

results[0]
from paradigm import Data, Time, MODE_SIM, MODE_OPT, pplot
import rhodium as rh
import numpy as np

bau = Data.load('dice_bau', lambda: Dice(mode=MODE_OPT).run())

nsow = 1000
import random
random.seed(1)
clim_sensi_sows = rh.LogNormalUncertainty('Climate Sensitivity', np.exp(ECS.mu), ECS.sigma).levels(nsow)

dc = Dice(time=Time(start=2015, end=2100, tstep=5), mode=MODE_SIM,
          setup={'S': bau.S, 't2xco2': clim_sensi_sows},
          default_sow=nsow, sow_setup={'Emission control rate GHGs': 0},
          vin=['MIU'], vout=['MAX_REL2C', 'MIN_NPVMITCOST'])

dcm = dc.asmodel()
dcm.responses.keys()
ds = rh.DataSet()

for sol in a:
    ds.append(dict(MIU=list(sol.variables), **dict(zip(dcm.responses.keys(), sol.objectives))))

rh.scatter2d(dcm, ds)
import pandas as pd
pd.DataFrame(ds['MIU']).T.plot()

rh.parallel_coordinates(dcm, ds)