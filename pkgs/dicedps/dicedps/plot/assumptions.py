from dicedps.plot.common import *

simtime = get_sim2plot(mtime, 100)

simtime.dc.run(np.zeros(47))

simtime.get('forcoth').plot()