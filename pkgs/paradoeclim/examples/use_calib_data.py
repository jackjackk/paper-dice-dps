import paradoeclim as dc
from dicedps.interactive import *




df = dc.get_calibration_data()
df.head()
dftrain = df.iloc[2:].copy()

df.plot(subplots=True)

plt.close('all')
sb.regplot(x='t2co',y='kappa', data=dftrain, order=3)

dftrain['t2coint'] = (dftrain['t2co']*100).astype(int)
df2 = dftrain.set_index('t2coint')
df2.head()
df3=df2.reindex(range(df2.index.min(),df2.index.max()+1)).interpolate(axis=0)

df3.iloc[::5].plot(x='t2co',y='kappa',kind='scatter')
dftrain.plot(x='t2co',y='kappa',kind='scatter')

import statsmodels.formula.api as smf
ret = smf.ols('kappa ~ t2co + pow(t2co, 2) + pow(t2co, 3) + pow(t2co, 4) + pow(t2co, 5) -1', data=df3.iloc[::3]).fit()
ret.summary()

dftrain['pred'] = ret.predict(dftrain['t2co'])
#plt.close('all')
for y in ['kappa','pred']:
    plt.scatter(dftrain['t2co'], dftrain[y])


ytgt = 'alpha'


ret = smf.ols(f'{ytgt} ~ t2co + pow(t2co, 2) + pow(t2co, 3) + pow(t2co, 4) + pow(t2co, 5) -1', data=df3.iloc[::3]).fit()
ret.summary()

dftrain['pred'] = ret.predict(dftrain['t2co'])
plt.close('all')
for y in [ytgt,'pred']:
    plt.scatter(dftrain['t2co'], dftrain[y])
