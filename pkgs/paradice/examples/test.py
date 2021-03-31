import logging

from paradigm import Data, MODE_OPT, MODE_SIM, Time

from paradice.dice import Dice
from paradigm.viz import pplot

logging.basicConfig(level=logging.DEBUG)

bau = Dice(mode=MODE_OPT)
bau.set_bau()
r = bau.solve()

r.TATM_year.plot()

pplot(r, Dice.plot_vlist_single)

"""
dsim = Dice(mode=MODE_SIM, endyear=2100, setup={'S':bau.optlrsav}, vin=['MIU'])
print(len(dsim.get_bounds()))
dsim.run([0.01]*len(dsim.get_bounds()))
print([o.eval(dsim) for o in dsim._objs])


a = dsim.run()
a.MIU.plot()
a.UTIL, a.TATM2100
dsim.set_bau().solve()
dsim.set('t2xco2',[2,bau.t2xco2,8])
dsim.get_bounds()
#d = Dice(mode=MODE_OPT)
#bau = d.set_bau().solve()
a = dsim.run(bau.MIU[1:], bau.S[:-4])
import pandas as pd
v='EIND'
b = pd.DataFrame({'opt':getattr(bau,v),'sim':getattr(a,v)[1]})
b.plot()
a.S.plot()
a.TATM.plot()


bau = DiceSolution.load('configurablemodel_20171013.dat')
dsim = Dice(mode=MODE_SIM, vin=['MIU','S']).set('S', bau)
print(d.get_bounds())
#d.run(np.r_[[bau.data[v].values for v in ['MIU','S']]])

"""
