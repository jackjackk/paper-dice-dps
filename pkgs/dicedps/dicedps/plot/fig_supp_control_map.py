from dicedps.dpsrules import MiuRBFController
from dicedps.plot.common import *

df = load_merged(orient='min')


df09mit = df[np.isclose(df[o.o_min_cbgemitcost_lab],0.9,atol=1e-4)]

apoint = df09mit.loc[mdps].sort_values(o.o_min_mean2degyears_lab).iloc[0]
amiu2run = v.get_x(apoint)
tpoint = df09mit.loc[mtime].sort_values(o.o_min_mean2degyears_lab).iloc[0]
tmiu2run = v.get_x(tpoint)

simdps = get_sim2plot(mdps, 1000)
simdps2 = get_sim2plot(mdps, 100)
simtime = get_sim2plot(mtime, 1000)
simtime2 = get_sim2plot(mtime, 1)

tpoint

dice = simdps.dc._mlist[1]
rbf = simdps.dc._mlist[0]

apoint
Tlist = np.linspace(1,5,101)
dTlist = np.linspace(-0.05,0.15, 101)

rbf.W[1:] = miu2run[:4]
rbf.Bs[0][1:] = miu2run[4:8]
rbf.Bs[1][1:] = miu2run[8:12]
rbf.Rs[0][1:] = miu2run[12:16]
rbf.Rs[1][1:] = miu2run[16:20]
rbf.TEMP_THRES = miu2run[20]
rbf.ABAT_RATE_AFTER_THRES = miu2run[21]


miuheat = {}
for t in tqdm(Tlist):
    for dt in tqdm(dTlist):
        rbf.TATM[2] = t
        rbf.MIU[2] = 0
        miuheat[(t,dt)]=float(MiuRBFController.rbf(rbf, 3, [t, t-dt])[0])


simdps.dc.run(amiu2run)
ytemp = simdps.get('TATM').round(3)
ymiu = simdps.get('MIU').round(2).mul(100)
ytemp.head()
ytempdiff = ytemp.diff().round(3)
df2scat = pd.concat([ytemp.stack(), ytempdiff.stack(), ymiu.stack()], keys=['temp','K/5yr','Abatement (%)'],axis=1).dropna().reset_index()

simtime2.dc.run(tmiu2run)



sb.set_context('notebook', font_scale=1.1)

clp()
fig = plt.figure(figsize=(12,4))

gs = GridSpec(1, 2)
ax_time = plt.subplot(gs[0, 0])
ax_cmap = plt.subplot(gs[0,1])

plot_var_cmap(simtime2, tmiu2run, yy=['MIU'], axs=[ax_time], barplot=False, annot_miu=False)
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax_cmap.transData, ax_cmap.transAxes)
trans2 = mtransforms.blended_transform_factory(ax_cmap.transAxes, ax_cmap.transData)
out = sb.scatterplot(x='temp', y='K/5yr', hue='Abatement (%)', data=df2scat, edgecolor=None, ax=ax_cmap, rasterized=True)
#out = sb.scatterplot(x='temp', y='Abatement (%)', hue='tempdiff', data=df2scat, edgecolor=None, ax=ax_cmap)
#ax_cmap.scatter(ytemp.loc[2020],ymiu.loc[2025], color='g')
ax_cmap.set_xlabel('Temperature (K)')
ax_cmap.set_ylabel('Change in Temperature (K/5yr)')
#hl, ll = ax_cmap.get_legend_handles_labels()
#ax_cmap.legend(hl, [f'{x:.0f}' for x in np.arange(-0.1,160,40)], title='Abatement (%)')


sb.despine(fig)

fig.tight_layout()

fig.savefig(inplot('fig-miu-30.pdf'))

clp()
fig = plt.figure(figsize=(12,6))


gs = GridSpec(1, 2)
ax_time = plt.subplot(gs[0, 0])
ax_cmap = plt.subplot(gs[0,1])

plot_var_cmap(simdps2, miu2run, yy=['MIU'], axs=[ax_time])
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax_cmap.transData, ax_cmap.transAxes)
hband = ax_cmap.fill_between([ytemp.loc[2020].min(), ytemp.loc[2020].max()], 0, 1, facecolor='0.5', alpha=0.5, transform=trans)
out = sb.scatterplot(x='temp', y='Abatement (%)', hue='K/5yr', data=df2scat, edgecolor=None, ax=ax_cmap, rasterized=True)
#ax_cmap.scatter(ytemp.loc[2020],ymiu.loc[2025], color='g')
ax_cmap.set_xlabel('Temperature (K)')
ax_cmap.set_ylabel('Change in Temperature (K/5yr)')
hl, ll = ax_cmap.get_legend_handles_labels()
ll = ['K/5yr',]+[f'{x:.1f}' for x in np.arange(-0.1,0.4,0.1)]
ax_cmap.legend(hl, ll)

hl
sb.despine(fig)

fig.tight_layout()

fig.savefig(inplot('fig-miu-50-full.pdf'))

hband.set_visible(False)

fig.savefig(inplot('fig-miu-40-noband.pdf'))

dfmiuheat=pd.Series(miuheat)
dfmiuheat.tail()

cmap = "YlGnBu"
fig, ax = plt.subplots(1,1,figsize=(6,6))
sb.heatmap(dfmiuheat.unstack().T,ax=ax,cmap=cmap,cbar=False)

dfmiuheat.head()