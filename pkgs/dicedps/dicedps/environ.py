import os
import sys
#os.chdir(join(home, 'working', 'dicedps', 'sandbox'))

project_dir = os.path.realpath(__file__)
root_dir = '/'.join(project_dir.split('/')[:-4])

inroot = lambda *x: os.path.join(root_dir, *x)
inscratch = lambda *x: inroot('output', *x)
incloud = lambda *x: inroot('archive', *x)
inoutput = lambda *x: incloud('output', *x)
indata = inoutput
indatabrick = lambda *x: indata('brick', *x)
indatadicedps = lambda *x: indata('moea', *x)
intestdir = lambda *x: inroot('tests', *x)
inplot = lambda *x: inroot('..', 'paper-dice-dps-overleaf', 'figures', *x)
inpaper = inplot

inbrick = lambda *x: inroot('pkgs', 'brick', *x)
indoeclim = lambda *x: inroot('pkgs', 'paradoeclim', 'paradoeclim', *x)
indicedps = lambda *x: inroot('pkgs', 'dicedps', 'dicedps', *x)
