from subprocess import Popen

import logging
logging.basicConfig(level=logging.INFO)
import sys

nseeds = 4

for i in range(nseeds):
    p = Popen([sys.executable, 'examples/simple_borg.py', str(i)])
