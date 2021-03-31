from collections import namedtuple, defaultdict
import string

from dicedps.dpsrules import MiuTemporalController
from dicedps.interactive import *
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec

import re
from matplotlib import ticker
from os.path import join

from paradicedoeclim.dicedoeclim import DiceDoeclim2

home = os.environ["HOME"]

# Name of the objectives set used for the optimization
last_oset = "v2"

ocolset = o.oset2labs[last_oset]

# Dataframes names
ldfthinned_2degyear_dps = 'dfthinned_2degyear_dps'
lsol_opt_1obj = 'sol_opt_1obj'
ldfsol1d_reeval = 'dfsol1d_reeval'
ldf_damfunc = 'df_damfunc'
ldf_csdist = 'df_csdist'
ldfthinned_mitcost = 'dfthinned_mitcost'
ldf_interp_mitcost = 'df_interp_mitcost'
ldf_thinned_diffs = 'df_thinned_diffs'
ldf_fig3_tseries = 'df_fig3_tseries'
lhpoints = 'hpoints'
lhpoints_mitcosts = 'hpoints_mitcosts'
lhpoints_temp = 'hpoints_temp'
lhpoints_miu = 'hpoints_miu'
lhpoints_df = 'hpoints_df'
lhpoints_dfdeep = 'hpoints_dfdeep'
lsims_ccs = 'sims_ccs'
ldfccs = 'dfccs'
ldfccs_miu = 'dfccs_miu'

ldffig2 = 'dffig2'

delta_cols = [o.o_min_q95damcost_lab,
              o.o_min_q95maxtemp_lab,
              o.o_min_miu2030_lab,
              o.o_min_miu2050_lab]
lytemp = 'temp'
lygross = 'ygross'
lyfinal = 'yfinal'
lytempdiff = '°C/5yr'
lydamcost ='damcost'
lyabatcost = 'abatcost'
lyloss = 'yloss'
lymiu = 'Abatement (%)'
ldam2 = ['Low (Nominal)', 'Medium', 'High']
ldamfunc_id = 'damfunc'
ldamfunc_lab = 'Damage function'
lcslevs = ['low', 'med', 'high']
lcslabs = ['Low', 'Medium\n(Nominal)', 'High']
cs2lab = dict(zip(lcslevs, [x.split('\n')[0] for x in lcslabs]))
lscencli_id = 'scen_cli'
lscencli_lab = 'Climate'
lsol_id = 'Solution w/ nominal\nmit. cost (% CBGE)'
labat = 'Abatement (%)'
labatmiu = 'Abatement strategy'
lyear = 'Year'
lsow = 'SOW'
lytemp_lab = 'Temperature (°C)'
y2lab = {lytemp: lytemp_lab}

f3xcol = o.o_min_cbgemitcost_lab
#scol = o.o_min_mean2degyears_lab
f3ycol = o.o_min_mean2degyears_lab
f3ycol = o.o_min_q95damcost_lab


def get_variable_from_model(dc: DiceDoeclim2, x: str, idmodel: int):
    try:
        ret = pd.DataFrame(
            np.array(getattr(dc._mlist[idmodel], x)[1:, :]),
            index=dc._mlist[idmodel].year[1:].tolist(),
        )
    except:
        ret = pd.Series(
            getattr(dc._mlist[idmodel], x)[1:, 0],
            index=dc._mlist[idmodel].year[1:].tolist(),
        )
    return ret


get_variable_from_doeclim = partial(get_variable_from_model, idmodel=3)
get_variable_from_dice = partial(get_variable_from_model, idmodel=1)
bget = get_variable_from_doeclim


def plot_variable_from_model(dc: DiceDoeclim2, x: str, idmodel: int, **kws):
    get_variable_from_model(dc, x, idmodel).plot(alpha=0.5, legend=False, **kws)


plot_variable_from_doeclim = partial(plot_variable_from_model, idmodel=3)
plot_variable_from_dice = partial(plot_variable_from_model, idmodel=1)
bplot = plot_variable_from_doeclim
aplot = plot_variable_from_doeclim

indata = lambda x: glob(incloud(os.path.join("data", x)))

prop_cycle = lambda: iter(plt.rcParams["axes.prop_cycle"])

prop_list = list(prop_cycle())

miumap = {
    "rbfTdT4": "DPS(T,dT|4)",
    "rbfTdT6": "DPS(T,dT|6)",
    "rbfT4": "DPS(T|4)",
    "rbfTdT1": "DPS(T,dT|1)",
    "time2": "Open loop",
}

orel2c = "Reliability 2C goal (% SOWs)"
outil = "Utility (% Consumption loss, CBGE)"
omit = "Mitigation cost\n(% Consumption loss, CBGE)"
odam = "Damage cost (% Consumption loss, CBGE)"
o2degy = "Two-degree years (°C-yr)"
omaxtemp = "95th-percentile max temperature (k)"

obj2lab2 = {
    o.o_max_rel2c_lab: orel2c,
    o.o_max_util_bge_lab: outil,
    o.o_min_loss_util_bge_lab: outil,
    o.o_min_cbgemitcost_lab: omit,
    o.o_min_cbgedamcost_lab: odam,
    o.o_min_mean2degyears_lab: o2degy,
    o.o_min_q95maxtemp_lab: omaxtemp,
    o.o_min_q95damcost_lab: "95th-percentile Damage cost\n(% Consumption loss, CBGE)",
    o.o_min_loss_util_95q_bge_lab: "95th-percentile Utility loss (% Consumption loss, CBGE)",
    o.o_min_loss_util_90q_bge_lab: "90th-percentile Utility loss (% Consumption loss, CBGE)",
    o.o_min_miu2020_lab: "Abatement in 2020",
    o.o_min_miu2030_lab: "Abatement in 2030",
    o.o_min_miu2050_lab: "Abatement in 2050",
}


olist = [orel2c, outil, omit, odam]

o2best = {orel2c: np.max, outil: np.min, omit: np.min, odam: np.min}

cdps = "Closed loop"
cdpstdt4 = "DPS(T,dT|4)"
cdpstdt6 = "DPS(T,dT|6)"
cdpstdt1 = "DPS(T,dT|1)"
cdpst4 = "DPS(T|4)"
copen = "Open loop"

mtime = "time2"
mdps = "rbfXdX41"
m2 = [mtime, mdps]

olist_greg4 = [obj for obj in o.oset2labs["greg4"]]

miumap2 = {"m" + x: y for x, y in miumap.items()}

miu_nice_order = ["Open loop", "DPS(T|4)", "DPS(T,dT|1)", "DPS(T,dT|4)"]

filename2miu = lambda f: miumap[os.path.basename(f).split("_")[1].split(".")[0]]


def get_flist_miulist(m):
    # didx = [miulist.tolist().index(m) for m in ]
    return flist, miulist


# BRICK

lcs = "Climate Sensitivity"
lod = "Ocean diffusivity"
las = "Aerosol scaling"

inbrickdir = lambda *x: os.path.join(os.environ["HOME"], "working", "brick", *x)


# dice_greg4b = lambda x: h.args2dice('-c doeclim -u 1 -w 10 -e 2250 -s 1 -o greg4b -C brick_fgiss_Tgiss_slognorm_o4 -t -r 4 -S 3 '+x)
dice_greg4b100 = lambda miu, *x: h.args2dice(
    "-c doeclim -u 1 -w 100 -e 2250 -s 1 -o greg4b -C brick_fgiss_Tgiss_slognorm_o4 -t -S 3 "
    + q.miu2arg(miu)
    + " ".join(x)
)
dice_greg4b = lambda miu, n, *x: h.args2dice(
    f"-c doeclim -u 1 -w {n} -e 2250 -s 1 -o greg4b -C brick_fgiss_Tgiss_slognorm_o4 -t -S 3 "
    + q.miu2arg(miu)
    + " ".join(x)
)
dice_last = lambda miu, n, *x: h.args2dice(
    f"-c doeclim -u 1 -w {n} -e 2250 -s 1 -o greg4f -C brick_fgiss_Tgiss_slognorm_o4 -t -S 3 "
    + q.miu2arg(miu)
    + " ".join(x)
)


def dice_cli(miu, n, cli, *x, obj_set=last_oset, **kwargs):
    sargs = f'-c doeclim -u 1 -w {n} -e 2250 -s 1 -o {obj_set} -C {cli} -t -S 3 {q.miu2arg(miu)} {" ".join(x)}'.strip()
    print(sargs)
    return h.args2dice(sargs, **kwargs)


var2lab = {
    "TATM": "Temperature (K)",
    "MIU": "Abatement (% base CO2)",
    "t2co": "Climate sensitivity (K)",
    "DAMFRAC": "Damage cost [% GDP/yr]",
    o.o_min_mean2degyears_lab: o.lab2short[o.o_min_mean2degyears_lab].replace(
        "\n", " "
    ),
}

var2ylim = {"TATM": [0, 5], "MIU": [0, 130]}

var2coeff = {"MIU": 100.0}

miu2lab = {
    "rbfXdX04": "Adaptive vanilla",
    "rbfXdX44": "Adaptive",
    "rbfXdX41": "Adaptive",
    "time2": "Non-adaptive",
"rbfXdX41_t1": "Adaptive, 5-yr update",
"rbfXdX41_t2": "Adaptive, 10-yr update",
"rbfXdX41_t3": "Adaptive, 15-yr update",
'rbfXdX41_ccs2050': 'Adaptive, CO2<0 after 2050',
'rbfXdX41_ccs2150': 'Adaptive, CO2<0 after 2150',
'time2_ccs2050': 'Non-adaptive, CO2<0 after 2050',
'time2_ccs2150': 'Non-adaptive, CO2<0 after 2150',
}


Sim2plot = namedtuple("Sim2plot", ["dc", "miu", "cmap", "colors", "get", "norm"])


def get_sim2plot(miu, nsow=100, cli="med", obj_set=last_oset, dice_kwargs=dict(), **kwargs):
    miu1 = miu
    dc = dice_cli(miu1, nsow, cli, *[f"--{x}={y}" for x, y in kwargs.items()], obj_set=obj_set, **dice_kwargs)
    # css1k = dc.t2co.sort_values()
    # css = css1k.iloc[::10]
    css = dc.t2co
    isorted = css.index
    norm = mpl.colors.Normalize(vmin=css.iloc[0], vmax=css.iloc[-1])
    cmap = mpl.cm.cool
    colors = cmap(MinMaxScaler().fit_transform(css.values.reshape(-1, 1), css).flatten())
    get = lambda x: getattr(dc, x).loc[:2200][isorted]
    return Sim2plot(dc=dc, miu=miu, cmap=cmap, norm=norm, get=get, colors=colors)


def plot_var_cmap(
    sim,
    amiu=None,
    yy=["MIU", "TATM"],
    years=[2020, 2050, 2100, 2150],
    axs=None,
    box=False,
    ylabel=True,
    barlabel=True,
    pad=0.05,
    oset2ret=last_oset,
    annot_miu=False,
    barplot=True,
    **kwargs,
):
    if isinstance(yy, str):
        yy = [yy]
    if axs is None:
        fig, axs = plt.subplots(len(yy), 1, figsize=(3, 2 * len(yy)), sharex=True)
    if isinstance(axs, mpl.axes.Axes):
        axs = [axs]
    if isinstance(amiu, pd.DataFrame):
        if len(amiu) > 1:
            amiu = amiu.loc[sim.miu]
            # assert len(amiu) == 1
        else:
            amiu = amiu.iloc[0]
        amiu = amiu[v.get_xcols(amiu)].dropna().values
    print(amiu)
    if amiu is not None:
        objs = sim.dc.run_and_ret_objs(amiu)
    for i, (ax, y) in enumerate(zip(axs, yy)):
        if barplot:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=pad)
            cb = mpl.colorbar.ColorbarBase(
                ax=cax, cmap=sim.cmap, norm=sim.norm, orientation="vertical"
            )
            cax.yaxis.set_ticks_position("right")
            if barlabel:
                # cax.text(1.2, .5, var2lab['t2co'], verticalalignment='center', rotation=90, transform=cax.transAxes)
                cb.set_label(var2lab["t2co"], rotation=-90, labelpad=10)
        if ylabel:
            ax.set_ylabel(var2lab.get(y, y))
        if amiu is not None:
            ydata = sim.get(y).mul(var2coeff.get(y, 1.0))
            ydata.plot(ax=ax, legend=False, color=sim.colors, **kwargs)
            print(objs)
            if (i == 0) and box:
                try:
                    sannot_temp = f"rel2c:  {objs[o.o_max_rel2c]:.0f}%"
                except:
                    sannot_temp = f"2cyear:  {objs[o.o_min_mean2degyears]:.1f}"
                sannot = f"{sannot_temp}\nmitcost:{objs[o.o_min_cbgemitcost]:.2f}%\ndamcost:{objs[o.o_min_cbgedamcost]:.2f}%"
                # sannot = amiu.name
                ax.annotate(
                    sannot,
                    xy=(0.99, 0.01),
                    xycoords="axes fraction",
                    size=8,
                    ha="right",
                    va="bottom",
                    bbox=dict(boxstyle="square,pad=0.3", fc="0.9", alpha=0.5, lw=0),
                )
        if y == "TATM":
            ax.axhline(2, ls="--", color="0.5")
        elif y == "MIU":
            if annot_miu:
                for year in years:
                    ycurr = ydata.loc[year].mean()
                    ax.annotate(f"{ycurr:.0f}%", xy=(year, ycurr))
        ax.set_ylim(var2ylim.get(y, None))
        ax.set_xlim([2015, 2175])
        ax.set_xticks(years)
        ax.set_xlabel("Year")
        for side in ["right", "top"]:
            ax.spines[side].set_visible(False)
    # fig.tight_layout()
    if amiu is not None:
        return v.get_o(
            pd.Series(
                {
                    o.obj2lab[x]: objs[x] if x[:3] == "MIN" else -objs[x]
                    for x in o.oset2vout[oset2ret]
                }
            )
        )
    return None


def smooth_amiu(amiu, years=None):
    # fig, ax = plt.subplots(1, 1)
    samiu = pd.Series(amiu, index=range(2015, 2250, 5))
    s1 = samiu.rolling(3, min_periods=0).median()
    s2 = samiu.rolling(3, min_periods=0).mean()
    samiu_smooth = pd.concat([s1, s2], axis=1).max(axis=1)
    # samiu.plot(ax=ax)
    # samiu_smooth.plot(ax=ax)
    return samiu_smooth


def plot_miu_cmap(
    simdemo,
    amiu=None,
    ahigh=None,
    chigh=None,
    axs=None,
    cbarticks=None,
    box=False,
    barlabel=True,
    pad=0.05,
    oset2ret=last_oset,
    cmap="viridis_r",
    cnorm=o.o_min_mean2degyears_lab,
    obj2bounds=None,
    **kwargs,
):
    yy = ["MIU"]
    if obj2bounds is None:
        obj2bounds = _obj2bounds
    if axs is None:
        fig, axs = plt.subplots(len(yy), 1, figsize=(3, 2 * len(yy)), sharex=True)
    if isinstance(axs, mpl.axes.Axes):
        axs = [axs]
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if cnorm is not None:
        norm = dict(vmin=obj2bounds[cnorm]["min"], vmax=obj2bounds[cnorm]["max"])
    if isinstance(norm, dict):
        norm = mpl.colors.Normalize(**norm)
    for i, (ax, y) in enumerate(zip(axs, yy)):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=pad)
        cb = mpl.colorbar.ColorbarBase(
            ax=cax, cmap=cmap, norm=norm, orientation="vertical"
        )
        cax.yaxis.set_ticks_position("right")
        if cbarticks is not None:
            cax.yaxis.set_ticklabels(cbarticks)
        if barlabel:
            # cax.text(1.2, .5, var2lab['t2co'], verticalalignment='center', rotation=90, transform=cax.transAxes)
            cb.set_label(var2lab[cnorm], rotation=-90, labelpad=10)
        ax.set_ylabel(var2lab.get(y, y))
        if amiu is not None:
            bmius = []
            amiux = amiu[v.get_xcols(amiu)]
            years = simdemo.dc.year.values[1:]
            for i in range(amiu.shape[0]):
                smiu = amiux.iloc[i].values.reshape(-1, 1)
                simdemo.dc._mlist[1].MIU[2:] = smiu
                simdemo.dc._mlist[1].X = smiu
                bmius.append(
                    [
                        float(MiuTemporalController.eqmiu_rule(simdemo.dc._mlist[1], t))
                        for t in range(1, 48)
                    ]
                )
            ydata = pd.DataFrame([smooth_amiu(bmiu, years) for bmiu in bmius]).T.mul(
                var2coeff[y]
            )  # , columns=range(2015, 2250, 5)).T.mul
            ydata.plot(
                ax=ax,
                legend=False,
                color=cmap(
                    scaler.fit_transform(
                        -amiu[cnorm].values.reshape(-1, 1), -amiu[cnorm]
                    ).flatten()
                ),
                **kwargs,
            )
        if ahigh is not None:
            bmius = []
            amiux = ahigh[v.get_xcols(ahigh)]
            smiu = amiux.values.reshape(-1, 1)
            simdemo.dc._mlist[1].MIU[2:] = smiu
            simdemo.dc._mlist[1].X = smiu
            bmius.append(
                [
                    float(MiuTemporalController.eqmiu_rule(simdemo.dc._mlist[1], t))
                    for t in range(1, 48)
                ]
            )
            ydata = pd.DataFrame([smooth_amiu(bmiu, years) for bmiu in bmius]).T.mul(
                var2coeff[y]
            )  # , columns=range(2015, 2250, 5)).T.mul
            ydata.plot(ax=ax, legend=False, color=chigh, lw=2)
        ax.set_ylim(var2ylim.get(y, None))
        ax.set_xlim([2015, 2175])
        ax.set_xticks([2020, 2050, 2100, 2150])
        ax.set_xlabel("Year")
        for side in ["right", "top"]:
            ax.spines[side].set_visible(False)
    # fig.tight_layout()
    return None


_obj2bounds = {
    o.o_min_mean2degyears_lab: {"min": -360, "max": 0},
    o.o_max_rel2c_lab: {"min": 0, "max": 36},
    o.o_max_util_bge_lab: {"min": -0.9, "max": -1.7},
    o.o_max_util_bge_lab: {"min": -0.9, "max": -1.7},
    o.o_min_cbgemitcost_lab: {"min": 0, "max": -1.3},
    o.o_min_cbgedamcost_lab: {"min": 0, "max": -1.3},
}


def oscale(olist, obj2bounds=None):
    """
    Fit `scaler` to given bounds.

    Parameters
    ----------
    olist
        List of objective labels
    obj2bounds
        Dict/pd.DataFrame w/ 1st dim = objective and 2nd dim = ['min','max']

    Returns
    -------
    scaler
    """
    if obj2bounds is None:
        obj2bounds = _obj2bounds
    y = np.array([[obj2bounds[y][w] for y in olist] for w in ["min", "max"]])
    series_units = (
        pd.Series([o.lab2short[x] for x in olist])
        .str.split("(", expand=True)
        .iloc[:, 1]
    )
    list_unique_units = series_units.unique()
    for uu in list_unique_units:
        uu_index = series_units[series_units == uu].index
        y[0, uu_index] = y[0, uu_index].min()
        y[1, uu_index] = y[1, uu_index].max()
    return MinMaxScaler().fit(y)  # .transform(x.values.reshape(1,-1))[0]


def xscale(x, scaler):
    return pd.Series(scaler.transform(x.values.reshape(1, -1))[0], index=x.index)


def get_scaled_df4plot_parallel(df, scaler):
    nobjs = len(df.index)
    return pd.DataFrame(scaler.transform(df.T).T, index=np.arange(nobjs))

ParallelPlotLayer = namedtuple('ParallelPlotLayer', ['data', 'kwargs'])

def plot_parallel_new(*pplist,
                      ax=None,
                      scaler=None,
                      obj2bounds=None,
                      iorder=None,
                      invert_min_labs=False,
                      keep_norm_labs=None,
                      arrwidth=1.0):
    nobjs = None
    if keep_norm_labs is None:
        keep_norm_labs = []
    toinvertlab = []
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for ipl, pl in enumerate(pplist):
        avec = pl.data.T
        if isinstance(avec, pd.Series):
            avec = [avec]
        if isinstance(avec, list):
            avec = pd.DataFrame(avec).T
        if iorder is None:
            iorder = range(len(avec.index))
        if obj2bounds is None:
            obj2bounds = avec.T.describe()
        if scaler is None:
            scaler = oscale(avec.index, obj2bounds)
        if nobjs is None:
            nobjs = len(avec)
            for i in np.arange(nobjs):
                ax.axvline(i, lw=1.5, color="0.5")
        if np.isnan(avec.values).all():
            continue
        scaled_df = get_scaled_df4plot_parallel(avec, scaler)
        kws = pl.kwargs.copy()
        color = kws.pop('color', 'viridis_r')
        labels = kws.pop('labels', None)
        if not 'lw' in kws:
            kws['lw'] = 2
        if not 'zorder' in kws:
            kws['zorder'] = ipl*10000
        if color in plt.colormaps():
            icolor = kws.pop('icolor', 0)
            cmap = plt.get_cmap(color)
            scaled_df = scaled_df.T.sort_values(by=icolor, ascending=False).T
            color = cmap(scaled_df.iloc[icolor])
        elif isinstance(color, str):
            color = [color]*avec.shape[1]
        if invert_min_labs:
            for i, s in enumerate(avec.index):
                if o.obj2asc[s]:
                    scaled_df.iloc[i, :] = 1-scaled_df.iloc[i, :]
                    toinvertlab.append(s)
        scaled_df.iloc[iorder].reset_index(drop=True).plot(ax=ax, legend=False, color=color, **kws)
        if labels is not None:
            for icount, (idsol, sol) in enumerate(scaled_df.iloc[iorder].T.iterrows()):
                ax.annotate(labels[icount],
                            xy=(0, sol.iloc[0]),
                            xytext=(-5, 0), textcoords='offset pixels',
                        va='center', ha='right')
    ax.set_xticks(range(nobjs))
    ax.xaxis.set_tick_params(length=0)
    ax2 = ax.twiny()
    ax.set_xlim([-0.2, nobjs - 1 + 0.1])
    ax.set_ylim([-0.05, 1.05])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(nobjs))
    ax2.xaxis.set_tick_params(length=0)
    xtickgen = lambda vec: [f"{(x):.1f}" for i, x in enumerate(vec)]
    xticklabs2 = []
    xticklabs = []
    xmins = xtickgen(scaler.data_min_)
    xmaxs = xtickgen(scaler.data_max_)
    draw_arrow_for_each_obj = False
    if (not np.any([o.obj2asc[s] for s in avec.index])):
        arrwidth = 1
        ax.arrow(
                -0.15,
                0.2,
                0.0,
                -0.1,
                width=0.02 * arrwidth,
                head_length=0.05,
                head_width=0.1 * arrwidth,
                color="0.5",
            )
    elif invert_min_labs or np.all([o.obj2asc[s] for s in avec.index]):
        ax.arrow(
            -0.15,
            0.8,
            0.0,
            0.1,
            width=0.02 * -arrwidth,
            head_length=0.05,
            head_width=0.1 * -arrwidth,
            color="0.5",
        )
    else:
        draw_arrow_for_each_obj = True

    for j, i in enumerate(iorder):
        s = avec.index[i]
        if s in toinvertlab:
            c = xmins[i]
            xmins[i] = xmaxs[i]
            xmaxs[i] = c
        if s in keep_norm_labs:
            xmins[i], xmaxs[i] = xtickgen([scaled_df.iloc[i,:].min(),
                                           scaled_df.iloc[i,:].max()])
        xticklabs2.append(f"{xmaxs[i]}")
        if j % 2:
            xticklabs.append(f"{xmins[i]}\n{o.lab2short[s]}")
        else:
            xticklabs.append(f"{xmins[i]}\n{o.lab2short[s]}")
        if draw_arrow_for_each_obj:
            if o.obj2asc[s]:
                ax.arrow(
                    j - 0.25,
                    -0.15 + 0.1,
                    0.0,
                    -0.05,
                    width=0.01 * arrwidth,
                    head_length=0.025,
                    head_width=0.05 * arrwidth,
                    color="0.5",
                    clip_on=False
                )
            else:
                ax.arrow(
                    j - 0.25,
                    -0.15 + 0.02,
                    0.0,
                    0.05,
                    width=0.01 * -arrwidth,
                    head_length=0.025,
                    head_width=0.05 * -arrwidth,
                    color="0.5",
                    clip_on=False
                )
    ax.set_xticklabels(xticklabs)
    ax2.set_xticklabels(xticklabs2)
    ax.tick_params(left=False)
    ax.tick_params(labelleft=False)
    for side in ["left", "bottom", "top", "right"]:
        for axx in [ax, ax2]:
            axx.spines[side].set_visible(False)


def plot_parallel(
    avec=None,
    ax=None,
    back=None,
    arrwidth=1.0,
    front=None,
    cfront=None,
        front_labels=None,
    cback=None,
    cargs={},
    iorder=None,
    scaler=None,
    obj2bounds=None,
    color="viridis_r",
    **kws,
):
    """

    Parameters
    ----------
    avec : pd.DataFrame
       rows = objectives, columns = solutions to plot
    ax : optional
        Matplotlib Axes to use, if None created
    back
    cback
        Color map or color
    iorder : optional
        index list to use for ordering objectives
    kws

    Returns
    -------

    """
    if avec is None:
        avec = pd.Series(np.nan, index=o.oset2labs["greg4e"])
    if iorder is None:
        iorder = range(len(avec.index))
    if obj2bounds is None:
        obj2bounds = avec.T.describe()
    if scaler is None:
        scaler = oscale(avec.index, obj2bounds)
    nobjs = len(avec)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for i in np.arange(nobjs):
        ax.axvline(i, lw=1.5, color="0.5")
    if back is not None:
        if isinstance(back, pd.DataFrame):
            scaled_df = get_scaled_df4plot_parallel(back)
            if cback is None:
                cback = "viridis"
            try:
                cmap = plt.get_cmap(cback)
                cback = cmap(scaled_df.iloc[0])
            except:
                pass
            scaled_df.plot(ax=ax, lw=2, legend=False, color=cback, **cargs)
        else:
            if cback is None:
                cback = ["0.5"] * len(back)
            for bvec, c in zip(back, cback):
                if not isinstance(c, dict):
                    c = {"color": c}
                print(c)
                ax.plot(np.arange(nobjs), xscale(bvec).values[iorder], lw=2, **c)
    if not np.isnan(avec.values).all():
        if isinstance(avec, pd.DataFrame):
            scaled_df = get_scaled_df4plot_parallel(avec)
            cmap = plt.get_cmap(color)
            if not "color" in kws:
                kws["color"] = cmap(scaled_df.iloc[0])
            scaled_df.plot(ax=ax, lw=2, legend=False, **kws)
        else:
            ax.plot(
                np.arange(nobjs), xscale(avec).values[iorder], lw=2, zorder=99999, **kws
            )
            navec = xscale(avec)
            for j, i in enumerate(iorder):
                x = avec.values[i]
                xn = navec.values[i]
                if not np.round(abs(x), 1) in [
                    np.round(abs(vec[i]), 1)
                    for vec in [list(scaler.data_min_), list(scaler.data_max_)]
                ]:
                    ax.annotate(s=f"{abs(x):.1f}", xy=(j, xn), va="bottom", ha="center")
    if front is not None:
        if isinstance(front, pd.DataFrame):
            scaled_df_front = get_scaled_df4plot_parallel(front)
            scaled_df_front.plot(ax=ax, lw=2, legend=False, zorder=99999, color=cfront)
            if front_labels is not None:
                for icount, (idsol, sol) in enumerate(scaled_df_front.T.iterrows()):
                    ax.annotate(front_labels[icount],
                                xy=(0, sol.iloc[0]),
                                xytext=(-5, 0), textcoords='offset pixels',
                            va='center', ha='right')
        else:
            ax.plot(
                np.arange(nobjs),
                xscale(front).values[iorder],
                lw=2,
                zorder=99999,
                color=cfront,
            )
            navec = xscale(front)
            for j, i in enumerate(iorder):
                x = front.values[i]
                xn = navec.values[i]
                if not np.round(abs(x), 1) in [
                    np.round(abs(vec[i]), 1)
                    for vec in [list(scaler.data_min_), list(scaler.data_max_)]
                ]:
                    ax.annotate(s=f"{abs(x):.1f}", xy=(j, xn), va="bottom", ha="center")
    ax.set_xticks(range(nobjs))
    ax2 = ax.twiny()
    if arrwidth>0:
        arrwidth = 1
        ax.arrow(
            -0.15,
            0.2,
            0.0,
            -0.1,
            width=0.02 * arrwidth,
            head_length=0.05,
            head_width=0.1 * arrwidth,
            color="0.5",
        )
    else:
        ax.arrow(
            -0.15,
            0.8,
            0.0,
            0.1,
            width=0.02 * -arrwidth,
            head_length=0.05,
            head_width=0.1 * -arrwidth,
            color="0.5",
    )
    ax.set_xlim([-0.2, nobjs - 1 + 0.1])
    ax.set_ylim([0, 1])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(nobjs))
    xtickgen = lambda vec: [f"{abs(x):.1f}" for i, x in enumerate(vec)]
    xticklabs2 = []
    xticklabs = []
    xmins = xtickgen(scaler.data_min_)
    xmaxs = xtickgen(scaler.data_max_)
    for j, i in enumerate(iorder):
        s = avec.index[i]
        xticklabs2.append(f"{xmaxs[i]}")
        if j % 2:
            xticklabs.append(f"{xmins[i]}\n{o.lab2short[s]}")
        else:
            xticklabs.append(f"{xmins[i]}\n{o.lab2short[s]}")
    ax.set_xticklabels(xticklabs)
    ax2.set_xticklabels(xticklabs2)
    ax.tick_params(left=False)
    ax.tick_params(labelleft=False)
    for side in ["left", "bottom", "top", "right"]:
        for axx in [ax, ax2]:
            axx.spines[side].set_visible(False)


def get_scaled_df(df2scale, df4scaling=None):
    if df4scaling is None:
        df = df2scale
    else:
        df = pd.concat([df2scale, df4scaling])
    ocols_cbge = [o.o_min_loss_util_bge_lab, o.o_min_cbgedamcost_lab, o.o_min_cbgemitcost_lab]
    mincost = df[ocols_cbge].min().min()
    maxcost = df[ocols_cbge].max().max()
    df4scaling = df.copy()
    lastrecord = df4scaling.iloc[-1].copy()
    for extremecost in [mincost, maxcost]:
        lastrecord = df4scaling.iloc[-1].copy()
        for ocol in ocols_cbge:
            lastrecord[ocol] = extremecost
        df4scaling = df4scaling.append(lastrecord, ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(df4scaling)
    ret = df2scale.copy()
    ret.loc[:, :] = scaler.transform(df2scale.values)
    return ret, scaler


miulist = [mdps, mtime]


def load_rerun(s=None, list_level_drop=["nfe"], orient="min"):
    if s is None:
        s = inoutput('dicedps', "*nfe5000000*v3*rerun.csv")
    return load_merged(s=s, list_level_drop=list_level_drop, orient=orient)


def load_merged(s=None, list_level_drop=["nfe"], orient="min"):
    """Load a set of '*(merged).csv' Pareto files as a DataFrame.

    Load matching files into a DataFrame.
    Invert sign of objectives if `orient`=='max'.
    Save orientation into name.
    Drop index levels of `list_level_drop`.

    Parameters
    ----------
    s : str, optional
        String in the pattern '*`s`*merged.csv' of file paths to be matched (default: `last_oset`).
    list_level_drop
        List of index levels to drop (by default "con", type of constraint, and "nfe", number of function evaluations).
    orient
        If max, invert values of objectives (default).
        If min, leave objectives untouched.

    Returns
    -------
    DataFrame
        Rows = solutions, columns = variables and objectives.

    """
    if s is None:
        s = inoutput('dicedps', f"*nfe5000000*{last_oset}*merged.csv")
    df = v.load_pareto(s, revert_max_objs=False)
    if orient == "max":
        olist = v.get_ocols(df)
        df[olist] = -df[olist]
    elif orient == "min":
        pass
    else:
        raise Exception(f"orient={orient} not accepted")
    df.index = df.index.droplevel(list_level_drop)
    df.name = orient
    return df


def plot_objective_pairs(df, orows=None, ocols=None,
                         axs=None, transpose=False,
                         prop_list=prop_list,
                         showideal=False):
    olist = v.get_ocols(df)
    if orows is None:
        orows = olist
    if ocols is None:
        ocols = olist
    if axs is None:
        fig, axs = plt.subplots(len(orows), len(ocols), figsize=(16, 8), dpi=150)
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    if axs.ndim < 2:
        if len(orows) == 1:
            axs = np.array([axs])
        elif len(ocols) == 1:
            axs = np.array([axs]).T
    for miu, p in zip([mdps, mtime], prop_list):
        dfcurr = df.loc[miu][olist]
        for i, row_obj in enumerate(orows):
            for j, col_obj in enumerate(ocols):
                curr_col_obj, curr_row_obj = col_obj, row_obj
                if transpose:
                    curr_col_obj, curr_row_obj = row_obj, col_obj
                axs[i, j].scatter(
                    dfcurr[curr_col_obj],
                    dfcurr[curr_row_obj],
                    s=1,
                    label=miu2lab[miu],
                    rasterized=True,
                    **p,
                )
                if showideal and (miu == mtime):
                    fbest = getattr(np, df.name)
                    hdiam = axs[i, j].scatter(
                        fbest(
                            dfcurr[curr_col_obj]
                        ),  # o.obj2fbest[col_obj](dfcurr[col_obj]),
                        fbest(
                            dfcurr[curr_row_obj]
                        ),  # o.obj2fbest[row_obj](dfcurr[row_obj]),
                        marker="D",
                        label=f"ideal",
                        color="k",
                    )
                xbnds = np.array(
                    [dfcurr[curr_col_obj].min(), dfcurr[curr_col_obj].max()]
                )  # np.array([obj2bounds[col_obj][x] for x in ['min','max']])
                xoff = max(abs(xbnds)) / 40
                ybnds = np.array(
                    [dfcurr[curr_row_obj].min(), dfcurr[curr_row_obj].max()]
                )  # np.array([obj2bounds[row_obj][x] for x in ['min', 'max']])
                yoff = max(abs(ybnds)) / 40
                axs[i, j].set_xlim([min(xbnds) - xoff, max(xbnds) + xoff])
                axs[i, j].set_ylim([min(ybnds) - yoff, max(ybnds) + yoff])
                if (j == 0) or transpose:
                    axs[i, j].set_ylabel(obj2lab2[curr_row_obj])
                if (i == 0) or transpose:
                    axs[i, j].set_xlabel(obj2lab2[curr_col_obj])
                if i + j == 0:
                    hextra = []
                    lextra = []
                    if showideal:
                        hextra.append(axs[i, j].scatter([], [], color="k", marker="D"))
                        lextra.append("Infeasible\nideal point")
                    handles = [
                        axs[i, j].scatter([], [], **p) for p in prop_list[:2]
                    ] + hextra
                    labels = [miu2lab[miu] for miu in [mdps, mtime]] + lextra
                    axs[i, j].legend(handles, labels)
                elif i == j:
                    axs[i, j].clear()


def get_sol(df, spec):
    ret = {}
    for miu in df.index.levels[0]:
        dfcurr = df.loc[miu].copy()
        dfnan = (np.nan) * dfcurr.iloc[[0]]
        for obj, v in spec.items():
            try:
                if v in ["min", "max"]:
                    dfcurr = dfcurr.sort_values(obj, ascending=(v == "min"))
                    v = dfcurr[obj].iloc[0]
                dfcurr = dfcurr[
                    np.isclose(dfcurr[obj], v, atol=o.obj2eps[o.lab2obj[obj]])
                ]
            except:
                print(f"Unable to slice {obj}: {v}")
                dfcurr = dfnan
        ret[miu] = dfcurr.iloc[0]
    return pd.concat(ret)


# Default argument keywords for sorting Pareto fronts
default_sort_pareto_kws = dict(
            by=[
                o.o_min_mean2degyears_lab,
                o.o_min_cbgedamcost_lab,
                o.o_min_cbgemitcost_lab,
                o.o_min_loss_util_bge_lab,
            ],
            ascending=True,
        )


def get_sol_by_query(
    df: pd.DataFrame, query=None, retout=False, sort_kws=default_sort_pareto_kws,
):
    """

    Parameters
    ----------
    df
        DataFrame of the form
                     dv0      dv1  ...  obj_xxx ... obj_yyy
        idsol
          jjj        ...
        ...
    mitcost
        Mitigation cost to filter with, or minimum in df if None
    relmax
        Sort in decreasing order if True
    retout
        Return tuple (x, objs) if True, otherwise only x
    atol
        Absolute tolerance to consider equal to `mitcost`
    sortby
        Sort by given columns before picking first element,
        otherwise take a mean

    Returns
    -------
    Either x vector of selected solution, or tuple with x
        and vector of objectives
    """
    if query is None:
        query = {
            o.o_min_cbgemitcost_lab: df[o.o_min_cbgemitcost_lab].min(),
        }
    y = df
    for k, v in query.items():
        if isinstance(v, tuple):
            v, atol = v
        else:
            atol = 1e-3
        y = y[np.isclose(df[k], v, atol=atol)]
    y = y.sort_values(**sort_kws)
    assert len(y) > 0
    try:
        groupby_indices = [x.name for x in y.index.levels]
        groupby_indices.remove("idsol")
        y = y.groupby(groupby_indices)
    except:
        print("unable to groupby")
        pass
    y = y.first()
    if retout:
        ret = (v.get_x(y), v.get_o(y))
    else:
        ret = y
    return ret


def get_best_1dim_sols(df, oset=last_oset, pre_sort_kws=default_sort_pareto_kws):
    """
    Find Pareto subset with single-objective best solutions.

    Parameters
    ----------
    df
      Single Pareto DataFrame
    oset
      Set of objectives for which to find best solutions
    pre_sort_kws
      Keywords for pre-sorting Pareto

    Returns
    -------
    pd.DataFrame
      Subset of df w/ best single-objective solutions
    """
    dps_best = []
    for obj in o.oset2labs[oset]:
        dps_best.append(
            df
                .sort_values(**pre_sort_kws)
                .sort_values(obj, ascending=o.obj2asc[obj])
                .iloc[[0]]
        )
    sol_opt_1obj: pd.DataFrame = pd.concat(dps_best).drop_duplicates()
    return sol_opt_1obj


def get_thinned_paretos(
    df: pd.DataFrame,
    thin_cols=o.o_min_mean2degyears_lab,
    thin_muls=2,
    thin_sort_kws=default_sort_pareto_kws,
    nroll=3,
):
    """
    Thin set of Paretos along given column by sorting, rounding its values and taking the first element of each group.

    Parameters
    ----------
    df
      Time and DPS Paretos DataFrame, e.g. loaded from `load_merged`
    thin_col
      Column to group by
    thin_ndec
      Resolution for the groupby column to be rounded to
    thin_sort_kws
      Dictionary to setup the sorting of each Pareto before grouping by

    Returns
    -------
    pd.DataFrame
      Thinned set of Paretos
    """
    try:
        groupby_indices = [x.name for x in df.index.levels]
        groupby_indices.remove("idsol")
    except:
        groupby_indices = []
    thin_cols_kws = {}
    if not isinstance(thin_cols, list):
        thin_cols = [thin_cols]
    if not isinstance(thin_muls, list):
        thin_muls = [thin_muls]
    thin_cols_labs = []
    for tcol, tmul in zip(thin_cols, thin_muls):
        thin_col_lab = f'thin_{tcol}'
        thin_cols_labs.append(thin_col_lab)
        thin_cols_kws[thin_col_lab] = df[tcol].mul(tmul).round().div(tmul)
    dfthinned = (df
                 .assign(**thin_cols_kws)
                 .set_index(thin_cols_labs, append=True)
                 .sort_values(**thin_sort_kws)
                 .groupby(groupby_indices+thin_cols_labs)
                 .first()
                 .sort_index(level=groupby_indices+thin_cols_labs))
    if nroll is not None:
        raise NotImplementedError()
        idx_thin_col = dfthinned.index.get_level_values('thin_col')
        #unit = pow(10,-thin_ndec)
        idx_full = range(idx_thin_col.min(), idx_thin_col.max()+1)
        unstack_indices = [x.name for x in dfthinned.index.levels]
        unstack_indices.remove('thin_col')
        dfthinned = (dfthinned
                 .unstack(unstack_indices)
                 .reindex(idx_full)
                 .interpolate(limit_area='inside')
                 .rolling(nroll, min_periods=1)
                 .mean()
                 .stack(unstack_indices)
                 .sort_index())
    print(f'Shape of thinned Pareto: {dfthinned.shape}')
    return dfthinned


def get_value_of_information(df,
                             index_by=o.o_min_mean2degyears_lab,
                             value_col=o.o_min_cbgemitcost_lab):
    """
    Compute VOI between DPS and time strategies.

    Parameters
    ----------
    df
      Paretos df
    index_by
      Column to use for indexing VOI
    value_col
      Column to use for measuring VOI (converting for mit cost)

    Returns
    -------
    pd.DataFrame
      VOI df
    """
    return (df
            .set_index(index_by, append=True)
            .loc[:, value_col]
            .unstack(level=0)
            .groupby(index_by)
            .first()
            .interpolate(axis=0)
            .diff(axis=1)[mtime]
            .dropna())


def get_sol_by_rel(
    df, rel=None, costmax=True, retout=False, atol=1e-3, sortby=o.o_min_cbgemitcost_lab
):
    if rel is None:
        rel = df[o.o_max_rel2c_lab].max()
    y = df[np.isclose(df[o.o_max_rel2c_lab], rel, atol=atol)]
    if sortby is not None:
        y = y.sort_values(o.o_max_rel2c_lab, ascending=(not costmax)).iloc[0]
    else:
        y = y.mean()
    assert len(y) > 0
    ret = v.get_x(y)
    if retout:
        ret = (ret, v.get_o(y))
    return ret


dpi4supp = 150

mm2inch = 1 / 25.4
w1col = 89 * mm2inch
w2col = 183 * mm2inch
w34col = (120 + 136) / 2.0 * mm2inch
hhalf = 247 / 2 * mm2inch


mpl.rcParams.update(
    {x: "small" for x in ["xtick.labelsize", "ytick.labelsize", "legend.fontsize"]}
)


# https://stackoverflow.com/a/39566040
SMALL_SIZE = 5
MEDIUM_SIZE = 6
BIGGER_SIZE = 7


mpl_nature = {
    "font.size": SMALL_SIZE,
    "axes.titlesize": BIGGER_SIZE,
    "axes.labelsize": MEDIUM_SIZE,
    "axes.linewidth": 0.5,
    "xtick.labelsize": SMALL_SIZE,
    "xtick.major.width": 0.5,
    "ytick.labelsize": SMALL_SIZE,
    "ytick.major.width": 0.5,
    "legend.fontsize": SMALL_SIZE,
    "figure.titlesize": BIGGER_SIZE,
    "lines.linewidth": 0.5,
}

mpl_interactive = {x: 2 * y for x, y in mpl_nature.items()}

# mpl.rcParams.update(mpl_interactive)
# mpl.rcParams.update(mpl_nature)


def savefig4paper(fig, s):
    # mpl.rcParams.update(mpl_nature)
    # fig.canvas.draw()
    fig.tight_layout()
    fig.savefig(inpaper(f"fig_{s}.pdf"))
    fig.savefig(inpaper(f"fig_{s}.png"), dpi=dpi4supp)
    # mpl.rcParams.update(mpl_interactive)
    # fig.canvas.draw()
    # fig.tight_layout()


lab_temp_year = lambda y: f"Temperature in {y} (K)"


def write_macro_generator(name: str):
    def write_macro(x: str, y: str, reset: bool = False):
        if reset:
            fspec = "w"
        else:
            fspec = "a"
        fnumbers = open(inpaper(f"fig_{name}.org"), fspec)
        fnumbers.write(f"#+MACRO: num-{x} {y}\n")
        fnumbers.close()

    return write_macro


def plot_scatter_temp_miu(df, axs, colors_abc, labels_abc, times=[2025, 2050, 2100]):
    for i, idsol in enumerate(df.index.levels[0]):
        for j, t in enumerate(times):
            df2scat = df.loc[idsol].loc[t].sort_values('°C/5yr')  # 'Abatement (%)', ascending=False)
            ax = axs[j]
            ax.set_ylim([0, 110])
            out = sb.scatterplot(x=lytemp,
                                 y=lymiu,
                                 size=lytempdiff,
                                 size_norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.31),
                                 data=df2scat,
                                 edgecolor='k',
                                 ax=ax,
                                 legend='brief',
                                 rasterized=True,
                                 alpha=0.7,
                                 color=colors_abc[i])
            ax.set_xlabel(f'Temperature in {t - 5} (°C)')
            ax.set_ylabel(f'Abatement\nin {t} (%)')
            if i == 2 and j == 2:
                hs, ls = ax.get_legend_handles_labels()
                print(ls)
                ax.legend(hs[1:5], [f'{float(x):.02}' for x in ls[1:5]], title='Change\nin temp.\n(°C/5yr)')
            else:
                ax.get_legend().remove()

    handles_abc = [plt.Line2D(range(1), range(1), markersize=8, color='white', marker='o', markeredgecolor='k',
                              markerfacecolor=x[:3], alpha=0.7, ls='-', lw=3) for x in colors_abc]

    axs[0].legend(handles_abc, labels_abc)



def reeval_sols_for_plot_temp_miu(df, simdps):
    df2scat = {}
    for i, (idsol, sol) in enumerate(df.iterrows()):
        simdps.dc.run(v.get_x(sol))
        ydict = {}
        ydict[lytemp] = simdps.get('TATM').round(3).shift(1)
        ydict[lygross] = simdps.get('YGROSS')
        ydict[lyfinal] = simdps.get('Y')
        ydict[lydamcost] = simdps.get('DAMFRAC') * 100.
        ydict[lyabatcost] = (simdps.get('ABATECOST') / ydict[lygross]) * 100.
        ydict[lyloss] = (1 - ydict[lyfinal] / ydict[lygross]) * 100.
        ydict[lymiu] = simdps.get('MIU').round(2).mul(100)
        ydict[lytempdiff] = ydict[lytemp].diff().round(3)
        # kws = prop_list[i]
        # ax.imshow(    axs[j].imshow(np.vstack((yyd,yyt)).T, cmap=cmap, aspect='auto', origin='bottom'))
        # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        # hband = ax.fill_between([ytemp.loc[2020].min(), ytemp.loc[2020].max()], 0, 1, facecolor='0.5', alpha=0.5,
        #                             transform=trans)
        df2scat[idsol] = pd.concat(ydict, axis=1).stack()
    return pd.concat(df2scat, names=['idsol', 't', 'sow'])