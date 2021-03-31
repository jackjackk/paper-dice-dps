from dicedps.interactive import *

inbrickdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'learn-brick', *x)

from paradoeclim import get_hist_forc_data, forc2comp
print('\n'.join(dfgreg.columns))
alpha = 1.
dfgreg = get_hist_forc_data()
dfgreg.loc[2003]
totgreg = dfgreg[forc2comp['forcing_nonaero']].sum(1) + \
          alpha * (dfgreg[forc2comp['forcing_aero']].sum(1))
totgreg.plot()

flnd = 0.29
dftony = pd.read_csv(inbrickdir('data', 'forcing_hindcast.csv'), index_col=0)
tottony = flnd * (dftony[['co2', 'nonco2.land', 'solar.land', 'volc.land']].sum(1) +
                  alpha * dftony['aerosol.land']) + \
          (1 - flnd) * (dftony[['co2', 'nonco2.ocean', 'solar.ocean', 'volc.ocean']].sum(1) +
                        alpha * dftony['aerosol.ocean'])

df=pd.concat([dfgreg,dftony],axis=1,join='inner')
dfc=df.corr() #.stack().abs().sort_values(ascending=False)
dfc.loc[dfgreg.columns,dftony.columns].stack().abs().sort_values(ascending=False)
istony = lambda x: (x=='co2') or ('.' in x)
for x,y in dfc[dfc<1.].iteritems():
    if istony(x[0]) and (not istony(x[1])):
        print(x, y)
pd.DataFrame({'greg': totgreg, 'tony': tottony}).plot()
clp()

dftony['aerosol.land'].div(dftony['aerosol.ocean']).describe()
dfgreg['ghg'].div(dfgreg['bc']).plot()
clp()
dftony[['aerosol.land','aerosol.ocean']].plot()
(tottony - totgreg).dropna().describe()

dfhad=pd.read_csv(inbrickdir('data','HadCRUT.4.6.0.0.annual_ns_avg.txt'),header=None,index_col=0,sep=' +')[1]
dfhad-dfhad.loc[1900].mean()

dfgreg.loc[2000:].plot()