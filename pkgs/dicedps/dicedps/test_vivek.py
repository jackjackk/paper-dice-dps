import logging
import re
import pandas as pd
import numpy as np

logger = logging.getLogger('read-borg-output')

def load_pareto_mpi(filename,  # path of BORG output
                    dvars,  # list of decision variable labels
                    objs,  # list of objective labels
                    objs2max=[],  # list of labels of objectives to maximize
                    ):
    logger.info(f'Processing "{filename}"')
    brecording = False
    last_lines = []
    last_props = {}
    pat = re.compile(r'//NFE=(\d+)')
    pat2 = re.compile(r'//(\w+)=(.+)$')
    nobjs = len(objs)
    ndvs = len(dvars)
    nfe = None
    data = None
    rets = {}
    rets2 = {}
    counter = 0
    with open(filename, 'r') as f:
        for line in f:  # read each line
            if line[0] in ['/', '#']:
                brecording = False
                if (nfe is not None) and (len(last_lines)>0):
                    ncols = ndvs + nobjs
                    nrows = len(last_lines)
                    data = np.zeros((nrows, ncols))
                    for i, l in enumerate(last_lines):
                        data[i, :] = [float(x) for x in l.split()]
                    ret = pd.DataFrame(data, columns=dvars + objs)
                    rets[nfe] = ret
                    last_props['NSOL'] = len(ret)
                    rets2[nfe] = pd.Series(last_props)
                    last_lines = []
                    last_props = {}
                if line[0] == '/':
                    try:
                        nfe = int(pat.match(line).groups()[0])
                    except:
                        mlist = pat2.match(line).groups()
                        last_props[mlist[0]] = float(mlist[1])
            else:
                if not brecording:
                    brecording = True
                    counter = 0
                last_lines.append(line)
                counter += 1
    ret = pd.concat(rets, names=['nfe','idsol'])
    ret2 = pd.concat(rets2, names=['nfe','variable'])
    logger.info('Inverting "max" objectives')
    for col in objs2max:
        ret[col] = -ret[col]
    return ret, ret2
