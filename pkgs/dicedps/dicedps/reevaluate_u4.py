from .dice_helper import get_uncertain_dicesim, obj2vout
from .viz_dps_policy import load_last_feather, get_xcols
from .dpsrules import MiuTemporalController

def main_reevaluate_u4():
    ds = load_last_feather()
    dss=ds[ds.obj=='simple2']
    dftemp = dss[dss.miu=='temporal/serial']

    dsimt = get_uncertain_dicesim(dps_class=MiuTemporalController, nunc=3, nsow=1000, vout=obj2vout['simple2'])
    dsimt.get_bounds()
    #dsimd = get_uncertain_dicesim(nunc=3, nsow=1000, vout=obj2vout['simple2'])
    dsimt._active_objs

    for idx, sol in dftemp.loc[:, get_xcols(dftemp)].iterrows():
        reobjs = dsimt.run(sol)
        dftemp.loc[idx,'robj0'] = reobjs[0]
        dftemp.loc[idx, 'robj1'] = reobjs[1]


if __name__ == '__main__':
    main_reevaluate_u4()