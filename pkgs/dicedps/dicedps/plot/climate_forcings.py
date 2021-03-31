from dicedps.plot.common import *

furb = inbrick('data', 'forcing_hindcast_urban.csv')
dfurb = pd.read_csv(furb, index_col=0)
flnd = 0.29
dfurb['aerosols'] = flnd*dfurb['aerosol.land'] + (1-flnd)*dfurb['aerosol.ocean']
dfurb['non-aerosols'] = (flnd * dfurb[['co2','nonco2.land','solar.land','volc.land']].sum(1) +
                         (1 - flnd) * dfurb[['co2','nonco2.ocean','solar.ocean','volc.ocean']].sum(1))

fgiss = inbrick('data', 'forcing_hindcast_giss.csv')
dfgiss = pd.read_csv(fgiss, index_col=0)
dfgiss['aerosols'] = dfgiss[['refa', 'aie', 'bc', 'snow']].sum(1)
dfgiss['non-aerosols'] = dfgiss[['ghg', 'o3', 'sh2o', 'stra', 'solar', 'land']].sum(1)


df=pd.concat([dfurb, dfgiss], keys=['urban','giss'], names=['source','year'], sort=True)[['aerosols','non-aerosols']]
#sb.factorplot(x='year',hue='source',col='level_2',y=0,data=df.reset_index())


def plot_hindcast_forcings():
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    for s, p in zip(['urban','giss'], prop_list):
        y = df.xs(s,0,'source')
        for ax, v in zip(axs, ['aerosols','non-aerosols']):
            ax.plot(y.index, y[v], lw=2, label=s, **p)
            ax.set_title(v)
            ax.set_xlabel('Year')
    axs[0].legend()
    axs[0].set_ylabel('Wm^-2')
    fig.tight_layout()
    fig.savefig(inplot('forcings.pdf'))


def extrap_forcings(bplot=False):
    from pyramid.arima import auto_arima
    _df = df.unstack(0)
    _df.index = pd.period_range(_df.index[0],_df.index[-1], freq='Y')
    df2extrap = _df

    _df=pd.read_csv(inbrick('data', 'HadCRUT.4.4.0.0.annual_ns_avg.txt'),
                   sep='\s+', index_col=0, header=None).iloc[:,0]
    _df.index = pd.period_range(_df.index[0],_df.index[-1], freq='Y')
    dftemp = _df

    yhat = {}
    yfut = {}
    fitted_model = {}
    #exo = dftemp.loc[df.index].values.reshape(-1,1)
    #exofit = exo[:-npred]
    #exopred = exo[-npred:]
    #exofut = dftemp.loc[df.index[-1]:2015].values.reshape(-1,1)
    exo = exofit = exopred = exofut = None
    for x, y in tqdm(df2extrap.items(), total=df2extrap.shape[1]):
        _df = pd.concat([y,dftemp], axis=1, keys=['forcing', 'temp'], join='inner').dropna()
        y = _df['forcing']
        ytemp = _df['temp']
        nfut = 2015 - y.index[-1].year
        npred = nfut
        exo = dftemp.loc[y.index].values.reshape(-1,1)
        exofut = dftemp.loc[(y.index[-1]+1):'2015'].values.reshape(-1,1)
        default_kws = dict(information_criterion='aic', #out_of_sample_size=npred,
                            start_p=0, start_q=0, maxiter=500, exogenous=exo,
                               max_p=5, max_q=5, m=1, trend='ct',
                               start_P=0, d=None, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
        #if 'non-aerosols' in x:
        #    default_kws['m'] = 11
        #    #default_kws.pop('start_params')
        y
        stepwise_model = auto_arima(y, **default_kws)
        fitted_model[x] = stepwise_model
        stepwise_model.fit(y.iloc[:-npred], exogenous=exo[:-npred])
        yhat[x] = pd.Series(stepwise_model.predict(n_periods=npred, exogenous=exo[-npred:]), index=y.index[-(npred):])
        stepwise_model.fit(y, exogenous=exo)
        yfut[x] = pd.Series(stepwise_model.predict(n_periods=nfut, exogenous=exofut), index=pd.period_range(y.index[-1]+1, 2015, freq='Y'))


    dfcomb = pd.concat([df2extrap, pd.DataFrame(yhat), pd.DataFrame(yfut)], axis=0, keys=['data','in-sample','out-sample']).stack([0,1])
    dfcomb.index.set_names(['type','year','v','source'], inplace=True)

    # Write extended forcings time series to be used with DICE
    forcpath = indicedps('data','forcings.csv')
    dffinal = pd.concat([df2extrap.sortlevel(axis=1),
                         pd.DataFrame(yfut).sortlevel(axis=1)], axis=0,
                        keys=['data','pred'], names=['type','year']).mean(axis=0,level='year')
    dffinal.stack('source').swaplevel().sortlevel(axis=0).to_csv(forcpath)

    if bplot:
        # Compare predictions vs observations
        fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
        for s, p in zip(['urban', 'giss'], prop_list):
            y = dfcomb.xs(s, 0, 'source')
            for ax, v in zip(axs, ['aerosols', 'non-aerosols']):
                for t, ls in zip(['data', 'in-sample', 'out-sample'], ['-', ':', '--']):
                    yy = y.xs(v, 0, 'v').xs(t, 0, 'type').loc['1990':]
                    ax.plot(yy.index.year, yy, lw=2, label=f'{s} ({t})', ls=ls, **p)
                ax.set_title(v)
        axs[0].legend()


"""
ret = pd.read_csv(forcpath, index_col=[0,1])
ret



dfpred.iloc[-npred:,:] = pd.DataFrame(yhat)[df.columns].values
dffut = df.copy().reindex(pd.period_range(df.index[0].year,2015,freq='Y'))
dffut.iloc[-nfut:,:] = pd.DataFrame(yfut)[df.columns].values
dfall = pd.concat([df,dfpred,dffut], keys=['data','pred','fut'], names=['source','year'])

dfall.loc['fut'].to_csv(fname.replace('.csv','_extrap.csv'))
sb.factorplot(x='year',data=dfall.stack().reset_index(),y=0,
              order=pd.period_range(1980,2015,freq='Y'),
              col='level_2',hue='source', col_wrap=4, n_boot=0, sharey=False, scale=0.7)
"""

if __name__ == '__main__':
    extrap_forcings()