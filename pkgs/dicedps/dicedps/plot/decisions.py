from mpl_toolkits.axes_grid1 import make_axes_locatable

from dicedps.plot.common import *

ds = xr.open_dataset(inrootdir('sandbox','last.nc'))
dc = {}
df = {}
colsmiu = {}
for miu, nfe in zip([mtime,mdps], [2000000,2000000]):
    df[miu] = ds.sel(nfe=nfe, miu=miu).to_dataframe().dropna(1, how='all').dropna(0)
for miu in [mtime, mdps]:
    dc[miu] = h.args2dice(f'{q.miu2arg(miu)} -o greg4 -u 1 -w 3 -e 2200')
    colsmiu[miu] = [f'dv{i}' for i in range(len(dc[miu].get_bounds()))]

def get_highest_rel2c_for_given_mitcost(x, y, col=o.o_max_rel2c_lab):
    return y[np.isclose(y[o.o_min_npvmitcost_lab], x, atol=1e-2)].sort_values(o.o_max_rel2c_lab, ascending=False).iloc[0].loc[col]

cols = np.array(sb.color_palette())[[3,0]]


#### GAINS ####
fig, ax = plt.subplots(1,1, figsize=(5,4))
xs = np.linspace(0.5,3.5,20)
reldps = np.zeros_like(xs)
reltime = np.zeros_like(xs)
for i in range(len(xs)):
    x = xs[i]
    reldps[i]=get_highest_rel2c_for_given_mitcost(x,df[mdps])
    reltime[i]=get_highest_rel2c_for_given_mitcost(x,df[mtime])
    #print(f'for mitcost={x}, reldps={reldps}, reltime={reltime}, gain={reldps/reltime-1}')
ax.plot(xs,reltime,color=cols[0],lw=2,label=copen)
ax.plot(xs,reldps,color=cols[1],lw=2,label=cdpstdt4)
ax.legend()
ax.set_xlabel(o.o_min_npvmitcost_lab)
ax.set_ylabel('Max Reliability of 2C for given mitigation cost')
sb.despine(fig)
fig.tight_layout()
fig.savefig(inplot('fig_gain_rel2c.png'), dpi=200)



### CONTROL TEMP ###
pstage = 1
fig, axs = plt.subplots(2,3, figsize=(8,5),sharey='row')
cslist = pd.Series(np.array(dc[miu]._mlist[3].t2co))
norm = mpl.colors.Normalize(vmin=cslist.iloc[0] - 0.5, vmax=cslist.iloc[-1])
cmap = mpl.cm.get_cmap('cool')
hs = {}
for x in [1,2,3]:
    ax = axs[0,x-1]
    ax.set_title(f'Mitigation cost\n{x}% GDP')
    if pstage > 1:
        ax.axhline(2, ls=':', color='0.5')
    axs[1,x-1].set_xlabel('Year')
    ax.set_xlim([2010.75, 2104.25])
    if x==1:
        ax.set_ylabel('Temperature increase [K]')
        ax.set_ylim([1,3.5])
        axs[1,x-1].set_ylabel('Abatement')
        axs[1, x - 1].set_ylim([0,1.1])
    for j, (miu, ls) in enumerate(zip([mtime,mdps], ['--','-'])):
        y = df[miu]
        miu2run = get_highest_rel2c_for_given_mitcost(x, y, col=colsmiu[miu])
        run = dc[miu].run(miu2run)
        for i, y in run.TATM.loc[:2100].T.iterrows():
            if pstage > 1 + j:
                if ((pstage == 3) and (((i>0) and (x==1)) or (x>1))) or (pstage==4):
                    hs[miu] = ax.plot(y.index, y.values, color=cmap(norm(cslist[i])), lw=2, ls=ls)
        for i, y in run.MIU.loc[:2100].T.iterrows():
            if miu == mtime:
                color = '0.5'
            else:
                color = cmap(norm(cslist[i]))
            if pstage > 0 + j*2:
                if ((pstage == 3) and (((i>0) and (x==1)) or (x>1))) or (pstage!=3):
                    hs[miu] = axs[1,x-1].plot(y.index, y.values, color=color, lw=2, ls=ls)
hopen = plt.Line2D((0,1),(0,0), color='k', ls='--', lw=2)
hdps = plt.Line2D((0,1),(0,0), color='k', ls='-', lw=2)
l = ax.legend([hdps, hopen], [cdps,copen])
#ax.legend(handles=[hs[mtime][0],hs[mdps][0]],labels=[copen,cdps])
sb.despine(fig)
fig.tight_layout(rect=[0,0,0.9,1])
if pstage > 1:
    cbar_im2a_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    cbar_im2a = mpl.colorbar.ColorbarBase(ax=cbar_im2a_ax, cmap=cmap,
                                   spacing='uniform',
                                   orientation='vertical',
                                   extend='neither')
    cbar_im2a.ax.get_yticks()
    cbar_im2a.set_ticks([0,0.5,1])
    cbar_im2a.ax.set_yticklabels([f'{x:.1f}' for x in run.t2co.values])
    cbar_im2a.ax.set_ylabel('Climate sensitivity (K)',rotation=-90, labelpad=15)
    cbar_im2a.ax.set_title('')
    cbar_im2a_ax.yaxis.set_ticks_position('left')
    cbar_im2a.get_clim()
plt.show()
fig.savefig(inplot(f'fig_control_temp{pstage}.pdf')) #, dpi=200)




fig, axs = plt.subplots(1,3, figsize=(6,2),sharey=True)
for miu in [mtime, mdps]:
    dc[miu] = h.args2dice(f'{q.miu2arg(miu)} -o greg4 -u 1 -w 3 -e 2200')
cslist = pd.Series(np.array(dc[miu]._mlist[3].t2co))
norm = mpl.colors.Normalize(vmin=cslist.iloc[0] - 0.5, vmax=cslist.iloc[-1])
cmap = mpl.cm.get_cmap('cool')
hs = {}
for x, ax in zip([1,2,3], axs):
    ax.set_title(f'Mitcost = {x}% GDP')
    ax.set_xlabel('Year')
    if x==1:
        ax.set_ylabel('Abatement')
    for miu, ls in zip([mtime,mdps], ['--','-']):
        y = df[miu]
        miu2run = get_highest_rel2c_for_given_mitcost(x, y, col=colsmiu[miu])
        run = dc[miu].run(miu2run)
        for i, y in run.MIU.loc[:2050].T.iterrows():
            if miu == mtime:
                color = '0.5'
            else:
                color = cmap(norm(cslist[i]))
            hs[miu] = ax.plot(y.index, y.values, color=color, lw=2, ls=ls)
hopen = plt.Line2D((0,1),(0,0), color='0.5', ls='--', lw=2)
hdps = plt.Line2D((0,1),(0,0), color='k', ls='-', lw=2)
ax.legend(handles=[hopen,hdps],labels=[copen,cdpstdt4])
sb.despine(fig)
fig.tight_layout()
fig.savefig(inplot('fig_control_miu.png'), dpi=200)

dice_bau = h.DiceBase(mode=h.MODE_OPT).set_bau().solve()
dcsingle = h.Dice(endyear=2025, mode=h.MODE_SIM, rsav=dice_bau)
rbf = r.MiuRBFController(dice=dcsingle, n=4)
miu=mdps
Tlist = np.linspace(1,5,101)
dTlist = np.linspace(-0.05,0.15, 101)
miuheat = {}
for x in [1,2,3]:
    miu2run = get_highest_rel2c_for_given_mitcost(x, df[miu], col=colsmiu[miu])
    for t in tqdm(Tlist):
        for dt in tqdm(dTlist):
            dcsingle.TATM[1]=t-dt
            dcsingle.TATM[2]=t
            miuheat[(x,t,dt)]=rbf.run(miu2run).MIU.loc[3]
miuheat
xtick_locator = ticker.MaxNLocator(6)
ytick_locator = ticker.MaxNLocator(11)
xform = ticker.FuncFormatter(lambda x, pos: f'{((Tlist[-1]-Tlist[0])*x/101+Tlist[0]):.1f}')
yform = ticker.FuncFormatter(lambda x, pos: f'{((dTlist[-1]-dTlist[0])*(x)/101+dTlist[0]):.2f}')

clp()
dfmiuheat=pd.Series(miuheat)
cmap = "YlGnBu"
fig, axs = plt.subplots(1,3,figsize=(7,3))
for x, ax in enumerate(axs):
    sb.heatmap(dfmiuheat.loc[x+1].unstack().T,ax=ax,cmap=cmap,cbar=False)
for ax in axs:
    ax.set_xlabel('Temperature (K)')
    #ax.set_ylim([-0.05,0.05])
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xform)
    ax.yaxis.set_major_locator(ytick_locator)
    if ax==axs[0]:
        ax.yaxis.set_major_formatter(yform)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
axs[0].set_ylabel('Change in Temp (K/5yr)')
fig.tight_layout(rect=[0,0,0.9,1.])
cbar_im2a_ax = fig.add_axes([0.9, 0.25, 0.025, 0.6])
cbar_im2a = mpl.colorbar.ColorbarBase(ax=cbar_im2a_ax, cmap=cmap,
                               spacing='uniform',
                               orientation='vertical',
                               extend='neither')
cbar_im2a_ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
cbar_im2a_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2f}'))
cbar_im2a.ax.set_title('      Abatement', fontsize=10, va='bottom')
cbar_im2a_ax.yaxis.set_ticks_position('right')
cbar_im2a.get_clim()
fig.savefig(inplot('fig_control_map.png'),dpi=200)

miuheat
for miu in [mtime, mdps]:
    dc[miu] = h.args2dice(f'{q.miu2arg(miu)} -o greg4 -u 1 -w 1000 -e 2200')
miu=mdps
for x in [1,2,3]:
    miu2run = get_highest_rel2c_for_given_mitcost(x, df[miu], col=colsmiu[miu])
    temp=dc[miu].run(miu2run).TATM.loc[:2195]
    dtemp = temp-temp.shift(axis=0)
    print('dT',(dtemp).min().min(), (dtemp).max().max())
    print('T',(temp).min().min(), (temp).max().max())
temp.min().min()
temp.max().max()
x=2
dtemp = temp-temp.shift(axis=0)
dtemp.max().max()
dtemp.T.describe().loc['max']>4
dc[miu]._mlist[3].time