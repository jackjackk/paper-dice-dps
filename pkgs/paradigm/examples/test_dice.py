from paradice.dice import Dice
import numpy as np
from paradigm import MODE_SIM

d = Dice(mode=MODE_SIM)

d.run(np.zeros(len(d.get_bounds())))
