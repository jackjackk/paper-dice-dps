from dicedps.plot.common import *

dc = h.args2dice('-c doeclim -u 1 -w 10 -e 2250 -s 1 -o greg4d -C high -t -S 1 -m time')

for i in range(1,4):
    dc._mlist[1].damfunc = i
    print(dc.run_and_ret_objs(0*np.ones(47)))

dc.C
dc.TATM.plot(legend=False)

#for f in glob('../dicedps/data/brick_*.nc'):
calibfiles = ['brick_fgiss_Tgiss_scauchy_o10',
 'brick_fgiss_Tgiss_scauchy_o4',
 'brick_fgiss_Tgiss_schylek_o10',
 'brick_fgiss_Tgiss_schylek_o4',
 'brick_fgiss_Tgiss_slognorm_o10',
 'brick_fgiss_Tgiss_slognorm_o4',
 'brick_fgiss_TgissOcheng_scauchy_o10',
 'brick_fgiss_TgissOcheng_scauchy_o4',
 'brick_fgiss_TgissOcheng_schylek_o10',
 'brick_fgiss_TgissOcheng_schylek_o4',
 'brick_fgiss_Thadcrut_scauchy_o10',
 'brick_fgiss_Thadcrut_scauchy_o4',
 'brick_fgiss_Thadcrut_schylek_o10',
 'brick_fgiss_Thadcrut_schylek_o4',
 'brick_fgiss_Thadcrut_slognorm_o10',
 'brick_fgiss_Thadcrut_slognorm_o4',
 'brick_fgiss_ThadcrutOcheng_scauchy_o4',
 'brick_fgiss_ThadcrutOcheng_schylek_o4',
 'brick_furban_Tgiss_scauchy_o10',
 'brick_furban_Tgiss_scauchy_o4',
 'brick_furban_Tgiss_schylek_o10',
 'brick_furban_Tgiss_slognorm_o10',
 'brick_furban_Tgiss_slognorm_o4',
 'brick_furban_TgissOcheng_scauchy_o10',
 'brick_furban_TgissOcheng_scauchy_o4',
 'brick_furban_TgissOcheng_schylek_o10',
 'brick_furban_TgissOcheng_schylek_o4',
 'brick_furban_Thadcrut_scauchy_o10',
 'brick_furban_Thadcrut_scauchy_o4',
 'brick_furban_Thadcrut_slognorm_o10',
 'brick_furban_Thadcrut_slognorm_o4',
 'brick_furban_ThadcrutOcheng_scauchy_o10',
 'brick_furban_ThadcrutOcheng_scauchy_o4']

_listcalibfiles = ['brick_fgiss_TgissOcheng_schylek_o4',
                   'brick_fgiss_TgissOcheng_schylek_o10',
                   'brick_fgiss_Tgiss_schylek_o4',
                   'brick_fgiss_ThadcrutOcheng_schylek_o4',
                   'brick_fgiss_Thadcrut_schylek_o4', 'brick_fgiss_Tgiss_schylek_o10',
                   'brick_furban_Tgiss_schylek_o10',
                   'brick_furban_TgissOcheng_schylek_o10',
                   'brick_furban_TgissOcheng_schylek_o4',
                   'brick_fgiss_Thadcrut_schylek_o10',
                   'brick_furban_TgissOcheng_scauchy_o4',
                   'brick_furban_ThadcrutOcheng_scauchy_o4',
                   'brick_furban_Tgiss_scauchy_o4',
                   'brick_furban_TgissOcheng_scauchy_o10',
                   'brick_furban_Thadcrut_scauchy_o4',
                   'brick_furban_Tgiss_slognorm_o4',
                   'brick_furban_Thadcrut_slognorm_o4',
                   'brick_furban_ThadcrutOcheng_scauchy_o10',
                   'brick_fgiss_TgissOcheng_scauchy_o4',
                   'brick_furban_Tgiss_slognorm_o10',
                   'brick_fgiss_ThadcrutOcheng_scauchy_o4',
                   'brick_furban_Tgiss_scauchy_o10',
                   'brick_fgiss_TgissOcheng_scauchy_o10',
                   'brick_fgiss_Tgiss_slognorm_o4',
                   'brick_furban_Thadcrut_slognorm_o10',
                   'brick_fgiss_Tgiss_scauchy_o4',
                   'brick_furban_Thadcrut_scauchy_o10',
                   'brick_fgiss_Thadcrut_slognorm_o4',
                   'brick_fgiss_Thadcrut_scauchy_o4',
                   'brick_fgiss_Tgiss_slognorm_o10',
                   'brick_fgiss_Thadcrut_slognorm_o10',
                   'brick_fgiss_Thadcrut_scauchy_o10',
                   'brick_fgiss_Tgiss_scauchy_o10']

len(_listcalibfiles)
_listcalibfiles[32]

csmed = {}
for f in calibfiles:
    a = u.get_sows_setup_mcmc(f, nsow=1000)
    csmed[f] = np.mean(a['setup']['t2co'])
pd.Series(csmed).sort_values().index.values


dc = dice_cli