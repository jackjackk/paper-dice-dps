import io
import string
from time import sleep

import numpy as np
import matplotlib.pylab as plt
from pandas import DataFrame, Panel
from paradigm.model import Model
from xarray import Dataset, DataArray
import logging
import matplotlib as mpl

colors = [list(np.array(list(bytes.fromhex(x['color'][1:])))/255) for x in mpl.rcParams['axes.prop_cycle']]


logger = logging.getLogger('paradigm:viz')


context2subplots = {
    'notebook': {'nrows': 2,
                 'ncols': None,
                 'figsize': (16, 9)},
    'paper': {'nrows': None,
              'ncols': 2,
              'figsize': (7, 9)}
}

def copy2clip(fig):
    try:
        from PyQt5.QtGui import QPixmap, QScreen, QImage
        from PyQt5.QtWidgets import QApplication, QWidget
        from matplotlib.backends.backend_qt5agg import FigureCanvas
        # sleep(2)
        # c = FigureCanvas(fig)
        if 'qt' == plt.get_backend()[:2].lower():
            fig.canvas.draw()
            pixmap = QWidget.grab(fig.canvas)
            QApplication.clipboard().setPixmap(pixmap)
        # else:
        #    buf = io.BytesIO()
        #    fig.savefig(buf)
        #    QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
    except Exception as e:
        logger.exception('exception')


def subplots(context, vlist):
    kws = context2subplots[context].copy()
    if kws['nrows'] is None:
        kws['nrows'] = int(np.ceil(len(vlist) / kws['ncols']))
    elif kws['ncols'] is None:
        kws['ncols'] = int(np.ceil(len(vlist) / kws['nrows']))
    return plt.subplots(**kws)

def _pplot_dataarray(y, ax=None):
    y.to_pandas().T.plot(ax=ax)

def pplot(y, vlist=None, context='notebook', bcopy2clip=False, upto=2100):
    plt.interactive(True)
    plt.style.use(f'seaborn-{context}')
    if isinstance(y, DataArray):
        fig, ax = plt.subplots(1,1)
        _pplot_dataarray(y, ax)
        return
    if isinstance(y, Dataset):
        if vlist is None:
            vlist = y.data_vars
        elif hasattr(vlist, 'plot_vlist_multi'):
            vlist = vlist = vlist.plot_vlist_multi
        nv = len(vlist)
        fig, axs = plt.subplots(1, nv, sharey=True)
        if nv == 1: axs = [axs]
        for i, (v, yv) in enumerate(y[vlist].data_vars.items()):
            _pplot_dataarray(yv, axs[i])
        return
    if not isinstance(y, list):
        y = [y]
    if hasattr(vlist, 'plot_vlist_single'):
        if len(y) == 1:
            vlist = vlist.plot_vlist_single
            # tlist = vlist.plot_tlist_single
        else:
            vlist = vlist.plot_vlist_multi
    else:
        if not isinstance(vlist, list):
            vlist = [vlist]
    fig, axs = subplots(context, vlist)
    for ax, vspec, abc in zip(axs.flat, vlist, string.ascii_uppercase):
        if isinstance(vspec, list):
            assert len(y) == 1
            setcycle = vspec
            ygetter = lambda v: getattr(y[0], f'{v}_year')
            ylabeller = lambda v: v
            axtitle = ''
        else:
            setcycle = y
            ygetter = lambda d: getattr(d, f'{vspec}_year')
            ylabeller = lambda d: d.name
            axtitle = vspec
        for i in setcycle:
            ycurr = ygetter(i)
            ycurr.ix[:upto].plot(ax=ax, label=ylabeller(i))
        ax.set_title(f'{abc}) {axtitle}')
        ax.xaxis.label.set_visible(False)
        ax.grid()
        ax.legend()
    fig.tight_layout()
    if bcopy2clip:
        copy2clip()