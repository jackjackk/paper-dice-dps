from paradice.dice import Dice
from paradigm import Time, Data, MODE_OPT, MODE_SIM
import rhodium as rh
import numpy as np
import matplotlib.pylab as plt


def main_test_borg():
    plt.interactive(True)
    bau = Data.load('dice_bau') #Dice(mode=MODE_OPT).run()

    nsow = 10
    ga0yr = np.power(1+(bau.ga0),1/5)-1
    ga0yr_sows = rh.NormalUncertainty('Output growth rate', mean=ga0yr, stdev=ga0yr*1.15/2.29).levels(nsow)
    import seaborn as sb
    ga0yr_sows
    sb.distplot(ga0yr_sows)
    dice_args = dict(time=Time(start=2015, end=2100, tstep=5), mode=MODE_SIM,
                     setup={'S': bau.S, 'ga0': np.power(1+ga0yr_sows, 5)-1},
                     default_sow=nsow)

    dsim = Dice(vin=['MIU'], sow_setup={'Emission control rate GHGs': 0,
                                        'Initial growth rate for TFP per 5 years': nsow}, **dice_args)

    a=dsim.run([0,]*17)
    plt.close('all')
    a.YGROSS_year.plot()
    ((pow(a.YGROSS_year.div(a.YGROSS_year.shift(1),0),1/5)-1).dropna().stack()).describe()


if __name__ == '__main__':
    main_test_borg()
