import logging
logging.basicConfig(level=logging.DEBUG)

from paradicedoeclim.cli import args2dice

dc = args2dice('-m temporal -C doeclim-medium-giss.nc -w 1')

"""
dc = DiceDoeclim2(MiuProportionalController)


from paradigm.misc import Timer
from paradice.dice import Dice
from paradicedoeclim.dicedoeclim import DiceDoeclim, DiceDoeclim2, MiuProportionalController
from paradigm.model import MODE_OPT, Data, Time, MODE_SIM
from paradigm.viz import pplot
import logging

logging.basicConfig(level=logging.INFO)



#bau = Dice(mode=MODE_SIM).d
#a = dc.run(MIU=bau.MIU_year.loc[2020:2105])
with Timer(): dc.run(X=[0.])
pplot(dc, Dice)


with Timer():
    for _ in range(2):
        pplot(dc, Dice)


pplot(dc, Dice)

plt.plot(dc._mlist[3].d.forcing_year, '-', color='r', label='dice')
plt.plot(dc._mlist[2].d.forcing_year, 'o', color='b', label='control')
plt.savefig('funny.pdf')

dc.FORC.plot()
t = Timer()
for _ in range(5):
    with t:
        b = dc.run(X=[0.01])
    t.reset()
dice.TATM_year.plot()
doeclim.temp_year.plot()
plt.show(block=True)
dice.MIU.div(dice.TATM)
dc.FORC.plot()
dc.forcing.plot()
dc.temp.plot()
dc.MIU.plot()

"""
