import string
from typing import Dict
import numpy as np
from dicedps.dice_helper import args2dice

dc = args2dice('-a borgpy -i 1 -c doeclim -u 1 -w 1000 -e 2250 -s 1 -o v2 -C med -t -O output/dicedps -m time -r 4 -x inertmax')

miu0 = np.zeros(47)
print(dc.run_and_ret_objs(miu0))




# region test mpi pareto file
from dicedps.plot.common import *
from dicedps.environ import *

frun = inscratch('dicedps', 'u1w1000doeclim_mtime_i1p400_nfe5000000_objv2_cinertmax_s1_seed0001_runtime.csv')
#frun_dps = inscratch('dicedps', 'u1w1000doeclim_mrbfXdX41_i1p400_nfe5000000_objv2_cnone_s3_seed0003_runtime.csv')
frun = 'scratch/moea/u1w1000doeclim_mtime_i1p200_nfe4000000_objgreg4d_cinertmax_s1_seed0001_runtime.csv'
frun = inscratch('dicedps', 'u1w1000doeclim_mtime_i1p400_nfe5000000_objv2_cinertmax_sX_seed000X_runtime.csv')
frun = inscratch('dicedps', 'u1w1000doeclim_mtime2_i1p400_nfe5000000_objv2_cnone_s1_seed0001_runtime.csv')
df, df2 = (v.load_pareto_mpi(frun,
                             keep_only_last=False,
                             metadata=True))
df2.xs('ElapsedTime', 0, 'variable')
#df2.unstack('variable')[['SBX','DE','PCX','SPX','UNDX','UM']].plot()
df[o.o_min_mean2degyears_lab].groupby('nfe').describe()

(2200-2015)/5+1
# endregion