import random
from paradice.dice import Dice, ECS
from paradigm import Time, MODE_SIM, Data, MODE_OPT, MultiModel
import rhodium as rh
import numpy as np
from borg4platypus import SerialBorgC
from .dpsrules import MiuRBFController


def main_test_dps_rbf():
    bau = Data.load('dice_bau', lambda: Dice(mode=MODE_OPT).run())

    random.seed(1)
    nsow=10
    clim_sensi_sows = rh.LogNormalUncertainty('Climate Sensitivity', np.exp(ECS.mu), ECS.sigma).levels(nsow)

    objlist = ['MAX_REL2C', 'MAX_UTIL', 'MIN_NPVMITCOST', 'MIN_NPVDAMCOST']
    epslist = [1e-1, 1e-1, 1e-4, 1e-4]
    vout = ['MAX_REL2C', 'MIN_NPVMITCOST']
    dice_args = dict(time=Time(start=2015, end=2100, tstep=5), mode=MODE_SIM,
                     setup={'S': bau.S, 't2xco2': clim_sensi_sows}, vout=vout,
                     default_sow=nsow)
    simdice2 = Dice(**dice_args)
    simdice2.set_inbound('MIU').set_outbound('TATM')
    controller2 = MiuRBFController(simdice2, default_sow=nsow)
    dc2 = MultiModel(controller2, simdice2)

    epsilons = [1e-1, 1e-4]
    borg_args = dict(epsilons=epsilons, liveplot=False, pbar=True,
                     log_frequency=max(10, 1000 // 100), name='test_rbf')
    algo = SerialBorgC(dc2.asproblem(), **borg_args)
    algo.run(1000)
    raw_results = [algo.result, ]
    import os
    os.environ['LD_LIBRARY_PATH']


    dc2.run([0.6896032024383553,0.703856244847145,0.5873999214308208,0.26120873389328103,11.619898594552627,8.767350876039673,1.8807352652961242,10.7535810642302,0.12853572031681715,14.101478056668183,12.015386616854624,0.440379628061871,-3.9101780966248465,-1.5197952945064284,12.530884023540665])
    dc2._mlist[1].d.MIU_year.plot()
    import matplotlib.pylab as plt
    plt.interactive(True)


if __name__ == '__main__':
    main_test_dps_rbf()