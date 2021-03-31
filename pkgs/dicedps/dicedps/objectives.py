import numpy as np

# Labels for single objectives
o_max_rel2c = 'MAX_REL2C'
o_min_mean2degyears = 'MIN_MEAN2DEGYEARS'
o_min_max2degyears = 'MIN_MAX2DEGYEARS'
o_min_npvmitcost = 'MIN_NPVMITCOST'
o_min_npvdamcost = 'MIN_NPVDAMCOST'
o_min_cbgemitcost = 'MIN_CBGEMITCOST'
o_min_cbgedamcost = 'MIN_CBGEDAMCOST'
o_min_q95damcost = 'MIN_Q95DAMCOST'
o_min_q95maxtemp = 'MIN_Q95MAXTEMP'
o_min_temp2100 = 'MIN_TATM2100'
o_min_miu2020 = 'MIN_MIU2020'
o_min_miu2030 = 'MIN_MIU2030'
o_min_miu2050 = 'MIN_MIU2050'
o_max_util = 'MAX_UTIL'
o_max_util_bge = 'MAX_UTIL_BGE'
o_min_loss_util_bge = 'MIN_LOSS_UTIL_BGE'
o_min_loss_util_95q_bge = 'MIN_LOSS_UTIL_95Q_BGE'
o_min_loss_util_90q_bge = 'MIN_LOSS_UTIL_90Q_BGE'
o_min_peakabatrate = 'MIN_PEAKABATRATE'

# Labels for sets of objectives
oset_simple2 = 'simple2'
oset_avg2 = 'avg2'
oset_avg3 = 'avg3'
oset_greg2 = 'greg2'
oset_greg3 = 'greg3'
oset_greg4 = 'greg4'
oset_greg4b = 'greg4b'
oset_greg4c = 'greg4c'
oset_greg4d = 'greg4d'
oset_v2 = 'v2'
oset_greg4e = 'greg4e'
oset_greg4f = 'greg4f'
oset_greg4g = 'greg4g'
oset_greg4h = 'greg4h'
oset_v3 = 'v3'
oset_v4 = 'v4'
oset_jack = 'jack'
oset_jack5 = 'jack5'
oset_jack6 = 'jack6'
oset_jack7 = 'jack7'
oset_all = 'all'

# Mapping between sets and objectives
oset2vout = {
    oset_simple2: [o_max_rel2c, o_min_npvmitcost],
    oset_avg2: [o_min_temp2100, o_min_npvmitcost],
    oset_avg3: [o_min_temp2100, o_min_npvmitcost, o_min_npvdamcost],
    oset_greg2:   [o_max_rel2c, o_max_util_bge],
    oset_greg3:   [o_max_rel2c, o_min_npvmitcost, o_min_npvdamcost],
    oset_greg4:   [o_max_rel2c, o_max_util_bge, o_min_npvmitcost, o_min_npvdamcost],
    oset_greg4b:  [o_max_rel2c, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost],
    oset_greg4c:  [o_max_rel2c, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_miu2020, o_min_miu2030, o_min_miu2050],
    oset_greg4d:  [o_min_mean2degyears, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost],
    oset_v2:  [o_min_mean2degyears, o_min_loss_util_bge, o_min_cbgemitcost, o_min_cbgedamcost],
    oset_greg4e:  [o_max_rel2c, o_min_mean2degyears, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost],
    oset_greg4f:  [o_min_mean2degyears, o_min_cbgemitcost, o_min_cbgedamcost, o_min_q95maxtemp],
    oset_greg4g:  [o_min_mean2degyears, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_miu2020, o_min_miu2030, o_min_miu2050, o_min_q95maxtemp],
    oset_greg4h:  [o_min_mean2degyears, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_miu2020, o_min_miu2030, o_min_miu2050, o_min_q95maxtemp, o_min_q95damcost],
    oset_v3:      [o_min_mean2degyears, o_min_loss_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_miu2020, o_min_miu2030, o_min_miu2050, o_min_q95maxtemp, o_min_q95damcost, o_min_loss_util_90q_bge, o_min_loss_util_95q_bge],
    oset_v4:      [o_min_mean2degyears, o_min_loss_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_miu2020, o_min_miu2030, o_min_miu2050, o_min_q95maxtemp, o_min_q95damcost, o_min_loss_util_90q_bge, o_min_loss_util_95q_bge, o_max_rel2c],
    #oset_greg5:   [o_max_rel2c, o_max_util_bge, o_min_cbgemitcost, o_min_cbgedamcost, o_min_q95damcost],
    oset_jack5:   [o_max_rel2c, o_max_util_bge, o_min_npvmitcost, o_min_npvdamcost, o_min_peakabatrate],
    oset_jack:    [o_max_rel2c, o_min_mean2degyears, o_min_npvdamcost, o_min_npvmitcost],
    oset_jack6:   [o_max_rel2c, o_max_util_bge, o_min_npvmitcost, o_min_npvdamcost, o_min_mean2degyears, o_min_peakabatrate],
    oset_jack7: [o_max_rel2c, o_max_util_bge, o_min_npvmitcost, o_min_npvdamcost, o_min_mean2degyears, o_min_max2degyears,
                 o_min_peakabatrate],
    oset_all: [o_max_rel2c, o_min_mean2degyears, o_min_max2degyears, o_min_npvmitcost, o_min_npvdamcost, o_min_cbgemitcost, o_min_cbgedamcost, o_min_temp2100, o_max_util, o_max_util_bge, o_min_peakabatrate],
}

B_LOWER_EPSS = True

# Epsilons
obj2eps = {
    o_max_rel2c: 1e-1,
    o_min_temp2100: 1e-2,
    o_min_q95maxtemp: 1e-2,
    o_min_miu2020: 1e-2,
    o_min_miu2030: 1e-2,
    o_min_miu2050: 1e-2,    
    o_max_util:  1e-1,
    o_max_util_bge: 1e-3,
    o_min_loss_util_bge: 1e-3,
    o_min_loss_util_90q_bge: 1e-3,
    o_min_loss_util_95q_bge: 1e-3,
    o_min_npvmitcost: 1e-3,
    o_min_npvdamcost: 1e-3,
    o_min_mean2degyears: 1e-1,
    o_min_max2degyears: 1e-2,
    o_min_peakabatrate: 1e-2,
    o_min_cbgedamcost: 1e-3,
    o_min_q95damcost: 1e-3,
    o_min_cbgemitcost: 1e-3,
}
if B_LOWER_EPSS:
	obj2eps.update({o_min_npvmitcost: 1e-4,
    				o_min_npvdamcost: 1e-4,
				    o_min_mean2degyears: 1e-2 })

oset2epss = {oset: [obj2eps[obj] for obj in objlist] for oset, objlist in oset2vout.items()}


# Labels
o_max_rel2c_lab = 'obj_Reliability_2C'
o_min_mean2degyears_lab = 'obj_Mean_Degree_Years_Above_2C'
o_min_max2degyears_lab = 'obj_Max_Degree_Years_Above_2C'
o_min_q95maxtemp_lab = 'obj_Min_q95_Max_Temperature'
o_min_q95damcost_lab = 'obj_Min_q95_Damage_Cost'
o_min_npvmitcost_lab = 'obj_Expected_NPV_Mitigation_Cost'
o_min_npvdamcost_lab = 'obj_Expected_NPV_Damage_Cost'
o_min_cbgemitcost_lab = 'obj_Expected_CBGE_Mitigation_Cost'
o_min_cbgedamcost_lab = 'obj_Expected_CBGE_Damage_Cost'
o_min_temp2100_lab = 'obj_Temperature_2100'
o_min_miu2020_lab = 'obj_Abatement_2020'
o_min_miu2030_lab = 'obj_Abatement_2030'
o_min_miu2050_lab = 'obj_Abatement_2050'
o_max_util_lab = 'obj_Expected_Utility'
o_max_util_bge_lab = 'obj_Expected_Utility_BGE'
o_min_loss_util_bge_lab = 'obj_Expected_Utility_Loss_BGE'
o_max_inv_loss_util_bge_lab = 'obj_Expected_Utility_BGE'
o_min_loss_util_95q_bge_lab = 'obj_q95_Utility_Loss_BGE'
o_min_loss_util_90q_bge_lab = 'obj_q90_Utility_Loss_BGE'
o_min_peakabatrate_lab = 'obj_Peak_Abatement_Rate'

obj2lab = {
    o_max_rel2c: o_max_rel2c_lab,
    o_min_mean2degyears: o_min_mean2degyears_lab,
    o_min_max2degyears: o_min_max2degyears_lab,
    o_max_util: o_max_util_lab,
    o_max_util_bge: o_max_util_bge_lab,
    o_min_loss_util_bge: o_min_loss_util_bge_lab,
    o_min_loss_util_95q_bge: o_min_loss_util_95q_bge_lab,
    o_min_loss_util_90q_bge: o_min_loss_util_90q_bge_lab,
    o_min_npvmitcost: o_min_npvmitcost_lab,
    o_min_npvdamcost: o_min_npvdamcost_lab,
    o_min_cbgemitcost: o_min_cbgemitcost_lab,
    o_min_cbgedamcost: o_min_cbgedamcost_lab,
    o_min_temp2100: o_min_temp2100_lab,
    o_min_miu2020: o_min_miu2020_lab,
    o_min_miu2030: o_min_miu2030_lab,
    o_min_miu2050: o_min_miu2050_lab,    
    o_min_peakabatrate: o_min_peakabatrate_lab,
    o_min_q95maxtemp: o_min_q95maxtemp_lab,
    o_min_q95damcost: o_min_q95damcost_lab,
}
lab2obj = {y:x for x,y in obj2lab.items()}
lab2short = {
    o_max_rel2c_lab: 'Reliability\n(% SOWs)',
    o_min_mean2degyears_lab: 'Warming above 2°C\n(°C-yr)',
    o_min_q95maxtemp_lab: '95th max temperature\n(K)',
    o_min_q95damcost_lab: '95th Damage cost\n(% CBGE)',
    o_max_util_bge_lab: 'Utility loss\n(% CBGE)',
    o_min_loss_util_bge_lab: 'Utility loss\n(% CBGE)',
    o_max_inv_loss_util_bge_lab: 'Utility \n(Normalized)',
    o_min_loss_util_95q_bge_lab: '95th Utility loss\n(% CBGE)',
    o_min_loss_util_90q_bge_lab: '90th Utility loss\n(% CBGE)',
    o_min_cbgedamcost_lab: 'Damage cost\n(% CBGE)',
    o_min_cbgemitcost_lab: 'Mitigation cost\n(% CBGE)',
}
oset2labs = {oset: [obj2lab[o] for o in vout] for oset,vout in oset2vout.items()}

obj2asc = {
    o_max_rel2c_lab: False,
    o_max_util_bge_lab: False,
    o_max_util_lab: False,
    o_min_cbgedamcost_lab: True,
    o_min_cbgemitcost_lab: True,
    o_min_loss_util_bge_lab: True,
    o_max_inv_loss_util_bge_lab: False,
    o_min_loss_util_95q_bge_lab: True,
    o_min_loss_util_90q_bge_lab: True,
    o_min_max2degyears_lab: True,
    o_min_mean2degyears_lab: True,
    o_min_miu2020_lab: True,
    o_min_miu2030_lab: True,
    o_min_miu2050_lab: True,
    o_min_npvdamcost_lab: True,
    o_min_npvmitcost_lab: True,
    o_min_peakabatrate_lab: True,
    o_min_q95damcost_lab: True,
    o_min_q95maxtemp_lab: True,
    o_min_temp2100_lab: True,
}

obj2fbest = { 
    o_max_rel2c_lab: np.max,
    o_min_mean2degyears_lab: np.min,
    o_max_util_bge_lab: np.max,
    o_min_npvmitcost_lab: np.min,
    o_min_npvdamcost_lab: np.min,
    o_min_cbgedamcost_lab: np.min,
    o_min_cbgemitcost_lab: np.min,
    o_min_peakabatrate_lab: np.min,
}

def labeller(objlabs=True):
    if objlabs:
        obj_labeller = lambda o: obj2lab[o]
    else:
        obj_labeller = lambda o: o
    return obj_labeller


def get_mpi_col2invert(objlabs=True):
    obj_labeller = labeller(objlabs)
    mpi_col2invert = {}
    for k, v in oset2vout.items():
        mpi_col2invert[k] = [obj_labeller(o) for o in v if o[:3].lower() == 'max']
    return mpi_col2invert


def normalize2min(df):
    ret = df.copy()
    for x, y in ret.items():
        if ('max' == x[:3].lower()) or ((x in lab2obj) and ('max'==lab2obj[x][:3].lower())):
            ret[x] = -y
    return ret


def oset2decs(oset):
    veps = oset2epss[oset]
    return -np.log10(veps).round(0).astype(int)


def oset2epsstring(oset):
    veps = oset2epss[oset]
    vdecs = -np.log10(veps).round(0).astype(int)
    vfmt = ['{:.%df}' % x for x in vdecs]
    return (','.join([xfmt.format(xeps) for xfmt, xeps in zip(vfmt, veps)]))
