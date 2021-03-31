import matplotlib as mpl
import seaborn as sb
import pandas as pd

df = pd.DataFrame({'x':[0,1,2],'y':[0,1,2]})

out = sb.scatterplot(x='x',
                     y='y',
                     size='y',
                     size_norm=mpl.colors.Normalize(vmin=0, vmax=0.1),
                     data=df2scat,
                     edgecolor='k',
                     ax=ax,
                     legend='',
                     rasterized=True)