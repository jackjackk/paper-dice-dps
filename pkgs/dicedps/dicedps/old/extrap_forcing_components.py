import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pylab as plt
import seaborn as sb
plt.interactive(True)
import os
inbrickdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'brick', *x)

def get_dfrcp():
    dfrcp_dict = {}
    for ircp in [26,45,6,85]:
        dfrcp_dict[f'{ircp}'] = pd.read_csv(inbrickdir('data',f'forcing_rcp{ircp}.csv'),index_col=0,parse_dates=True)
    dfrcp=pd.concat(dfrcp_dict) #.stack() #.reset_index()
    years = dfrcp.index
    dfrcp.index = pd.period_range(years[0], years[-1], freq='Y')
    dfrcp['sum_nonco2'] = dfrcp[['nonco2','aerosol.direct','aerosol.indirect','volcanic','solar','other']].sum(axis=1)
    dfrcp['aero'] = dfrcp[['aerosol.direct','aerosol.indirect']].sum(1)
    return dfrcp

def get_furb(startyear = 1880):
    furb = '../paradoeclim/paradoeclim/data/forcing_hindcast_urban.csv'
    dfurb = pd.read_csv(furb, index_col=0).loc[startyear:]

def get_fgiss(startyear = 1880):
    fgiss = '../paradoeclim/paradoeclim/data/forcing_hindcast_giss.csv'
    dfgiss = pd.read_csv(fgiss, index_col=0).loc[startyear:]
    dfgiss['aero'] = dfgiss[['refa', 'aie', 'bc', 'snow']].sum(1)


dfrcp = get_dfrcp()
dfurb = get_furb()
dfgiss = get_fgiss()


# Plot RCP components
dfrcp_tidy = dfrcp.stack().reset_index()
dfrcp_tidy.columns = ['rcp','year','forc','value']
sb.factorplot(x='year',y='value',hue='rcp',sharey=False,
              col_order=['co2','sum_nonco2','total'],col='forc',col_wrap=5,
              size=5,data=dfrcp_tidy, scale=0.5, ci=None)


# Plot sum of nonco2 range
dfrcp['sum_nonco2'].unstack(0).loc[2015:2100].describe().T[['min','mean','max']].plot()


# Compare recent past aerosol between RCP range, Urban and GISS
rcpcols = ['aerosol.direct','aerosol.indirect']
urbancols = [[f'aerosol.{x}'] for x in ['land','ocean']]
pd.concat([dfrcp[rcpcols].sum(1).unstack(0), #.describe().T[['min','mean','max']],
           (0.29)*dfurb[urbancols[0]].sum(1)+(1-0.29)*dfurb[urbancols[1]].sum(1),
           dfgiss['aero']],axis=1).loc[2000:2200].plot()

dfrcp.columns
dfdf['co2'] + df['nonco2.land']

from pyramid.arima import auto_arima
df.index = pd.period_range(df.index[0],df.index[-1], freq='Y')
fitted_model = {}
yhat = {}
yfut = {}
npred=5
nfut = 2015-df.index[-1].year
#exo = dftemp.loc[df.index].values.reshape(-1,1)
#exofit = exo[:-npred]
#exopred = exo[-npred:]
#exofut = dftemp.loc[df.index[-1]:2015].values.reshape(-1,1)
exo = exofit = exopred = exofut = None

for x, y in tqdm(df.items(), total=df.shape[1]):
    default_kws = dict(information_criterion='oob', out_of_sample_size=npred,
                        start_p=0, start_q=0, maxiter=500, exogenous=exo,
                           max_p=5, max_q=5, m=1, trend='c',
                           start_P=0, d=None, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    if 'solar' in x:
        default_kws['m'] = 11
        #default_kws.pop('start_params')
    if ('volc' in x) or ('aerosol' in x) or (x in ['land','o3','stra']):
        yhat[x] = y.iloc[-npred]*np.ones(npred)
        yfut[x] = y.iloc[-1]*np.ones(nfut)
        continue
    stepwise_model = auto_arima(y, **default_kws)
    fitted_model[x] = stepwise_model
    yhat[x] = stepwise_model.fit_predict(y.iloc[:-npred], n_periods=npred, exogenous=exopred)
    yfut[x] = stepwise_model.fit_predict(y, n_periods=nfut, exogenous=exofut)

dfpred = df.copy()
dfpred.iloc[-npred:,:] = pd.DataFrame(yhat)[df.columns].values
dffut = df.copy().reindex(pd.period_range(df.index[0].year,2015,freq='Y'))
dffut.iloc[-nfut:,:] = pd.DataFrame(yfut)[df.columns].values
dfall = pd.concat([df,dfpred,dffut], keys=['data','pred','fut'], names=['source','year'])

dfall.loc['fut'].to_csv(fname.replace('.csv','_extrap.csv'))
sb.factorplot(x='year',data=dfall.stack().reset_index(),y=0,
              order=pd.period_range(1980,2015,freq='Y'),
              col='level_2',hue='source', col_wrap=4, n_boot=0, sharey=False, scale=0.7)


g=sb.FacetGrid(df.stack().reset_index(),col='level_1',col_wrap=4)
g.map(plt.scatter, 'year', 0, s=5)
### TO REMOVE ###
df=pd.read_csv('forcing_hindcast_giss.csv', index_col=0)
dftidy = df.stack().reset_index()
dftidy.columns = ['year','forc','value']
dftidy['diff'] = df.diff().stack(dropna=False).values
dftidy

dftemp = pd.read_csv('giss_temp_anomalies_1880_2017.csv', skiprows=1, index_col=0)['J-D']
import seaborn as sb
g = sb.FacetGrid(data=dftidy, col='forc', col_wrap=4)
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
a=seasonal_decompose(df['ghg'])
g.map(plt.scatter, 'year', 'value')
g.map(sm.graphics.tsa.plot_acf, 'diff', lags=40)

df['solar'].iloc[::11].plot()

yfut
dfall
yhat = pd.Series(stepwise_model.predict(n_periods=len(ytest)), index=ytest.index)
pd.DataFrame({'data':y, 'pred':yhat}).plot()
print(stepwise_model.aic())
