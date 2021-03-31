from dicedps.uncertainties import ncbrick2pandas, get_sows_setup_mcmc
from dicedps.dice_helper import args2climcalib
import pandas as pd


cli_list = ['low', 'med', 'high']
cli2lab = {
    'low': 'Chylek (2008)',
    'med': 'PALEOSENS (2012)',
    'high': 'Urban (2010)'
}
dfnc = {}
for cli in cli_list:
    ncfile = args2climcalib(cli)
    sset = get_sows_setup_mcmc(ncfile, nsow=1000)
    sows = pd.DataFrame(sset['setup'])
    sows.columns = list(sset['sow_setup'].keys())
    dfnc[cli2lab[cli]] = sows.describe()
dfall: pd.DataFrame = pd.concat(dfnc)


print(dfall.stack().unstack(1).swaplevel().sort_index().to_latex(float_format='%.2f'))