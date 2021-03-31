from paradice.dice import Dice
from paradigm.model import MODE_OPT
from paradigm.viz import pplot
import logging
logging.basicConfig(level=logging.DEBUG)
bau = Dice(mode=MODE_OPT).set_bau().solve(tee=True)
opt = Dice(mode=MODE_OPT).solve()

pplot(opt, Dice)
pplot([bau,opt], Dice)
