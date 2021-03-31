from dicedps.plot.common import *

inbrickdir = lambda *x: os.path.join(os.environ['HOME'], 'working', 'brick', *x)

dfold=pd.read_csv(inbrickdir('data','gouretski_ocean_heat_3000m.txt'),
               sep='\s+', skiprows=1, index_col=0,
               names=['year','oheat','stddev'])
dfold.index = pd.period_range(1953, 1996, freq='Y')
dfold['oheat'] = dfold['oheat'].subtract(dfold.loc['1960':'1990','oheat'].mean())

def txtcheng2dfyear(f):
    df = pd.read_csv(inbrickdir('data', f), sep='\s+', comment='%')
    df.index = pd.period_range(f'{df["year"].iloc[0]}-{df["month"].iloc[0]}',
                               f'{df["year"].iloc[-1]}-{df["month"].iloc[-1]}', freq='M')
    # Add columns
    if 'OHC700-2000m' in df.columns:
        df['oheat'] = df[['OHC700-2000m', 'OHC0-700m']].sum(1)
        df['sigma2'] = ((df['Error, 95% CI is OHC+/-Error'] / 2) ** 2 +
            (df['Error, 95% CI is OHC+/-Error.1'] / 2) ** 2)
    elif 'Errorbar, lower bound,95%CI' in df.columns:
        lheat = 'Ocean-Energy 10^22 Joules'
        llow = 'Errorbar, lower bound,95%CI'
        lup = 'Errorbar, upper bound, 95%CI'
        sigma2_from_errbars = lambda x: (max(x[lheat] - x[llow], x[lup] - x[lheat]) / 2)**2
        df['sigma2'] = df.apply(sigma2_from_errbars, axis=1)
        df['oheat'] = df[lheat]
    # Yearly average weighted on monthly number of days
    df['ndays'] = df.index.to_series().dt.days_in_month
    df = df.resample('Y').apply(lambda x: np.average(x, weights=df.loc[x.index, 'ndays']))
    # Add stddev
    df['stddev'] = np.sqrt(df['sigma2'])
    # Normalize oheat
    df['oheat'] = df['oheat'].subtract(df.loc['1960':'1990','oheat'].mean())
    # Remove last year
    df = df.iloc[:-1]
    return df

dftoa = txtcheng2dfyear('TOA_OHC_errorbar_1940_2015_2.txt')

dfiap = txtcheng2dfyear('IAP_OHC_estimate_update.txt')

dfall = pd.concat([df['stddev'] for df in [dfold,dftoa,dfiap]], keys=['old','toa','iap'], names=['source','year'])
dfall.unstack().T.plot()

# Compare old and new OHC datasets
clp()
fig, ax = plt.subplots(1,1,figsize=(12,6))
for df, lab, p in zip([dfold,dftoa], ['Gouretski (2007)','Cheng (2017)'], prop_list):
    ax.fill_between(df.index.year, df['oheat']-df['stddev'], df['oheat']+df['stddev'], alpha=0.5, **p)
    ax.plot(df.index.year, df['oheat'], lw=2, label=lab, **p)
ax.set_ylabel('Ocean heat anomaly [10^22 J]')
ax.legend()
ax.set_xlabel('Year')
fig.tight_layout()
fig.savefig(inplot('ocean.pdf'))

# Write CSV file
(dftoa[['oheat','stddev']]
 .reset_index()
 .to_csv(inbrickdir('data','cheng_ohc.txt'), sep=' ', index=False,
 header=['% year', 'heat-anomaly (10^22 J)', 'std.dev. (10^22 J)']))

