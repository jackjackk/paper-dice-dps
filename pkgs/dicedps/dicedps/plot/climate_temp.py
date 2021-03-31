from dicedps.plot.common import *

inbrickdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'brick', *x)

dfhc=pd.read_csv(inbrickdir('data','HadCRUT.4.4.0.0.annual_ns_avg.txt'),
               sep='\s+', index_col=0, header=None).iloc[:,0]
dfhc.index.name = 'year'
dfhc.name = 'temp'

dfhcerr=pd.read_csv(inbrickdir('data','HadCRUT.4.4.0.0.annual_ns_avg_realisations','HadCRUT.4.4.0.0.annual_ns_avg.1.txt'),
               sep='\s+', index_col=0, header=None).iloc[:,1]
dfhcerr.index.name = 'year'
dfhcerr.name = 'stddev'

dfhc = pd.concat([dfhc, dfhcerr], axis=1)

dfgi=pd.read_csv(inbrickdir('data','GLB.Ts+dSST.csv'),
               sep='\s+', na_values='***', index_col=0,
                 usecols=[0,13,19], names=['year','temp','stddev2'])
dfgi['stddev'] = dfgi['stddev2']/2

fig, ax = plt.subplots(1,1,figsize=(12,6))
for df, lab, p in zip([dfhc,dfgi], ['HadCRUT','NASA GISS'], prop_list):
    df['temp2'] = df['temp'] - (df['temp'].loc[1880:1900].mean())
    ax.fill_between(df.index, df['temp2']-df['stddev'], df['temp2']+df['stddev'], alpha=0.5, **p)
    ax.plot(df.index, df['temp2'], lw=2, label=lab, **p)
ax.legend()
ax.set_ylabel('Temperature anomaly [K wrt 1880-1900]')
ax.set_xlabel('Year')
fig.tight_layout()
fig.savefig(inplot('temp.pdf'))

