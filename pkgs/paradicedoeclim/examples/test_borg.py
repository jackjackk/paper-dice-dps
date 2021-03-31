from borg4platypus import SerialBorgC
from paradice.dice import Dice
from paradigm import Data, Time, MODE_SIM, MODE_OPT

bau = Data.load('dice_bau', lambda: Dice(mode=MODE_OPT))
dc = Dice(endyear=2100, mode=MODE_SIM, vin=['MIU'], vout=['MAX_REL2C', 'MAX_UTIL'],
          setup={'S': bau.S}, default_sow=1000,
          sow_setup={'Emission control rate GHGs': 0})

#pplot([bau, dc.d], 'S')
dcp = dc.asproblem()
max_nfe = 500
log_freq = max(1,int(max_nfe/100))
epss = 0.01

import os

# algo = ExternalBorgC(dcp, epsilons=epss,
#                           log_frequency=log_freq,
#                           name='dctest', mpirun='-np 2')
algo = SerialBorgC(dcp, epsilons=epss,
                          log_frequency=log_freq,
                          name='dctest',seed=1)
algo.run(max_nfe)
