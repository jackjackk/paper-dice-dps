from paradice.dice import Dice, ECS
from paradigm import Timer, logging, pplot, MODE_SIM, Time, Data, MODE_OPT, MultiModel
import numpy as np
from paradicedoeclim.dicedoeclim import DiceDPS, MiuProportionalController, MiuPolyController
import rhodium as rh

logging.basicConfig(level=logging.INFO)

bau = Data.load('dice_bau', lambda: Dice(mode=MODE_OPT).run())
nsow = 10
import random
random.seed(1)
clim_sensi_sows = rh.LogNormalUncertainty('Climate Sensitivity', np.exp(ECS.mu), ECS.sigma).levels(nsow)
objlist = ['MAX_REL2C', 'MIN_NPVMITCOST', 'MAX_UTIL', 'MIN_NPVDAMCOST']
dice_args = dict(time=Time(start=2015, end=2100, tstep=5), mode=MODE_SIM,
                 setup={'S': bau.S, 't2xco2': clim_sensi_sows},
                 default_sow=nsow, vout=objlist[:2])
simdice = Dice(**dice_args)
simdice.set_inbound('MIU').set_outbound('TATM')
controller = MiuPolyController(simdice, default_sow=nsow)
dc = MultiModel(controller, simdice)

dc.run(-2,2,2,2,2)
dc.MIU.plot()
dc._vout
dc.TATM.T.describe()
dc.TATM.max().max()
a=dc.TATM-dc.TATM.shift()
a.max().max()
(a.max().max())**2
"""
dd = DiceDPS(MiuProportionalController, default_sow=1000)
pdice = dd.asproblem()
dd.run_and_ret_objs_list(1.2)
pdice.function(1.2)

ddsingle = DiceDPS(MiuProportionalController)
dd.dice.set(t2xco2=np.linspace(2,4.5,10))
dd.MIU
dd.t2xco2
dd.lam
with Timer(): dd.run(X=[0.1])

with Timer():
    for _ in range(1000):
        ddsingle.run(X=[0.1])

dd.MIU.plot()

pplot(dd, Dice)


"""