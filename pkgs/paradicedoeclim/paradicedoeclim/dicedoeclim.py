from paradice.dice import DiceBase, Ctrl
from paradoeclim.doeclim import Doeclim
from paradigm.model import BiModel, MODE_SIM, Data, MODE_OPT, Time, Model, MultiModel, VAR, EQUATION, RANGESET
import numpy as np
import pandas as pd

class DiceDoeclim(BiModel):

    _v = 2

    def dice2doeclim(self, t1, t2):
        dice = self._m1
        doeclim = self._m2
        tcurr_dice = t1
        tcurr_doeclim = t2  # t in doeclim corresponding to t in dice
        tprev_dice = max(1, tcurr_dice-1)
        fcurr_dice = dice.FORC[tcurr_dice]
        fprev_dice = dice.FORC[tprev_dice]
        ycurr_dice = dice.year[tcurr_dice]
        yprev_dice = dice.year[tprev_dice]
        ycurr_doeclim = doeclim.year[tcurr_doeclim]
        if ycurr_doeclim == ycurr_dice:
            doeclim.forcing[tcurr_doeclim] = fcurr_dice
            return [0,1]
        assert ycurr_doeclim < ycurr_dice
        if fcurr_dice == fprev_dice:
            doeclim.forcing[tcurr_doeclim] = doeclim.forc_data.loc[ycurr_doeclim].sum()
        else:
            doeclim.forcing[tcurr_doeclim] = fprev_dice + (fcurr_dice-fprev_dice)*(ycurr_doeclim-yprev_dice)/(ycurr_dice-yprev_dice)
        return [1]

    def doeclim2dice(self, t1, t2):
        dice = self._m1
        doeclim = self._m2
        ycurr_dice = dice.year[t1]
        ycurr_doeclim = doeclim.year[t2]
        if ycurr_dice == ycurr_doeclim:
            dice.TATM[t1] = doeclim.temp[t2]
            return [0,1]
        assert ycurr_doeclim < ycurr_dice
        return [1]

    def __init__(self):
        self._time_doeclim = Time(start=1900, end=2101)
        self._time_doeclim_hist = Time(start=1900, end=2015)
        self._time_doeclim_run = Time(start=2015, end=2101)
        # Build doeclim simulator
        """
        doeclimhistpath = f'dicedoeclim_doeclimhist_v{DiceDoeclim._v}'
        self._hist = Data.load(doeclimhistpath,
                               lambda: (Doeclim(time=Time(start=1900,end=2015,tstep=1),mode=MODE_SIM)
                                        .run()))
                                        """
        simdoeclim = Doeclim(time=self._time_doeclim, mode=MODE_SIM)
        self._hist = simdoeclim.run(time=self._time_doeclim_hist)
        simdoeclim.set_inbound('forcing').set_outbound('temp')
        #setup = {x.lower() + '0': {None: getattr(self._hist, x)[self._hist.year == 2014].values[0]}
        #         for x in Doeclim.state0},

        # Build dice simulator
        self._time_dice = Time(start=2015,end=2105,tstep=5)
        baupath = 'dicedoeclim_dicebau_v{vdd}'.format(vdd=DiceDoeclim._v)
        self._bau = Data.load(baupath, lambda: DiceBase(mode=MODE_OPT).set_bau().solve())
        simdice = DiceBase(self._time_dice, mode=MODE_SIM, vin=['MIU'],
                       setup={'S': self._bau.S_year[:self._time_dice.end].values,
                              'fex0': {None: -self._bau.FORC[1]+self._bau.forcoth[1]+
                                             self._hist.forcing_year[2015]}},
                       inbound=['TATM'], outbound=['FORC'])
        self._one2two = self.dice2doeclim
        self._two2one = self.doeclim2dice

        #self._counter = 0

        super().__init__(simdice, simdoeclim)

    def run(self, *args):
        ret = super().run(*args, kwargs2={'time':self._time_doeclim_run,'eval_eqs_notime':False})
        #self._m1.d.save(f'debug_fe{self._counter:03d}')
        #self._counter += 1
        return ret


class DiceDoeclimAdapter(Model):

    def __init__(self, dice, doeclim, time=None):
        assert dice._default_sow == doeclim._default_sow
        if time is None:
            time = dice.time
        self._time_doeclim = Time(doeclim.time.start, doeclim.time.end, tstep=doeclim.time.tstep)
        self._tdice2doeclim = {}
        self._year2tdoeclim = {}
        self._year2tdice = {}
        for tdice, yr in enumerate(dice.time._range.year.values):
            tdoeclim = np.where(doeclim.time._range.year.values==yr)[0][0]
            self._year2tdoeclim[yr] = tdoeclim+1
            self._year2tdice[yr] = tdice+1
            self._tdice2doeclim[tdice+1] = tdoeclim+1
        #self._tdoeclim2dice = [0,] + [int(x) for x in (np.floor((np.subtract(time._range.year.values, dice.time.start)/dice.time.tstep))+1)]
        self._diceyears = dice.time._range.year.values
        self._dice = dice
        self._doeclim = doeclim
        super().__init__(time, mode=MODE_SIM, default_sow=dice._default_sow)
        self.set_inbound('temp', 'FORC').set_outbound('forcing', 'TATM')

    def _body(self):
        self.FORC = self._dice.FORC #self.new(VAR, self._dice.t, doc='Forcing from DICE')
        self.forcing = self._doeclim.forcing #self.new(VAR, self._dice.t, doc='Forcing for DOECLIM')
        self.TATM = self._dice.TATM #self.new(VAR, self._dice.t, doc='Temp for DICE')
        self.temp = self._doeclim.temp #new(VAR, self._dice.t, doc='Temp from DOECLIM')

    def eqforc_rule(self, m, t):
        tcurr_dice = t
        ycurr_dice = self._dice.year[tcurr_dice]
        tprev_dice = max(1, tcurr_dice - 1)
        yprev_dice = self._dice.year[tprev_dice]
        tcurr_doeclim = self._tdice2doeclim[t]
        #doeclim.forcing[tcurr_doeclim] = doeclim.forc_data.loc[ycurr_doeclim].sum()
        fcurr_dice = self.FORC[tcurr_dice]
        fprev_dice = self.FORC[tprev_dice]
        if yprev_dice == ycurr_dice:
            self.forcing[tcurr_doeclim] = fcurr_dice
            #elif fcurr_dice == fprev_dice:
            #    doeclim.forcing[tcurr_doeclim] = doeclim.forc_data.loc[ycurr_doeclim].sum()
        else:
            for tcurr_doeclim in range(self._tdice2doeclim[tprev_dice]+1, self._tdice2doeclim[tcurr_dice]+1):
                ycurr_doeclim = self._doeclim.year[tcurr_doeclim]
                self.forcing[tcurr_doeclim] = fprev_dice + (fcurr_dice - fprev_dice) * (ycurr_doeclim - yprev_dice) / (
                                                       ycurr_dice - yprev_dice)
                if ycurr_doeclim<ycurr_dice:
                    self.add_collateral_output(ycurr_doeclim, 'forcing')
        #if np.isnan(doeclim.forcing[tcurr_doeclim]):
        #    pass
        return None

    def eqtemp_rule(self, m, t):
        t1 = self._tdice2doeclim[t]
        assert not np.array(np.isnan(self._doeclim.temp[t1])).any()
        self._dice.TATM[t] = self._doeclim.temp[t1]
        return None

    def _body_eqs(self):
        # Wait for DICE FORC
        self.eqdiceforc = self.new(EQUATION, self.FORC, None)
        # Write DOECLIM forcing
        self.eqdoeclimforc = self.new(EQUATION, self.forcing, self.eqforc_rule)
        # Wait for DOECLIM temp
        self.eqdoeclimtemp = self.new(EQUATION, self.temp, None)
        # Write DICE TATM
        self.eqdicetemp = self.new(EQUATION, self.TATM, self.eqtemp_rule)




class DiceDPS(MultiModel):

    def __init__(self, controlclass, **kwargs):

        self._time_dice = Time(start=2015, end=2100, tstep=5)
        bau = Data.load('dice_bau', lambda: Dice(time=self._time_dice, mode=MODE_OPT).solve())
        simdice = Dice(time=self._time_dice, mode=MODE_SIM,
                       setup={'S': bau.S_year[:self._time_dice.end].values},
                       vout=['MAX_REL2C', 'MIN_NPVMITCOST'], **kwargs)
        #simdice.setfixed(simdice.TATM, 1, False)
        simdice.set_inbound('MIU').set_outbound('TATM')

        controller = controlclass(simdice, **kwargs)

        self.dice = simdice
        self.control = controller

        super().__init__(controller, simdice)


class DiceDoeclim2(MultiModel):

    def __init__(self, controlclass, mode=MODE_SIM, startyear=1880, endyear=2100, dice_kwargs=None, doeclim_kwargs=None, control_kwargs=None, dice_bau=None, **kwargs):
        if dice_kwargs is None:
            dice_kwargs = {}
        if doeclim_kwargs is None:
            doeclim_kwargs = {}
        if control_kwargs is None:
            control_kwargs = {}
        dice_kwargs.update(kwargs)
        doeclim_kwargs.update(kwargs)
        # Doeclim
        self._time_doeclim = Time(start=startyear, end=endyear)
        doeclim_kwargs['setup']['temp0'][:] = 0
        simdoeclim = Doeclim(time=self._time_doeclim, mode=mode, **doeclim_kwargs)
        self._time_doeclim_pre = Time(start=startyear, end=1921)
        self._pre = simdoeclim.run(time=self._time_doeclim_pre)
        # Normalize temperature to be zero between 1880 and 1920
        simdoeclim.temp0[:] = -self._pre.temp_year.loc[1880:1920].mean()
        # Run history
        self._time_doeclim_hist = Time(start=startyear, end=2015)
        self._hist = simdoeclim.run(time=self._time_doeclim_hist)
        assert self._hist.temp_year.loc[1880:1920].mean().abs().max() < 1e-12

        simdoeclim.set_inbound('forcing').set_outbound('temp')
        self._time_doeclim_run = Time(start=2015, end=endyear)

        # Dice
        self._time_dice = Time(start=2015,end=endyear,tstep=5)
        try:
            self._bau = dice_kwargs['bau']
        except:
            self._bau = Data.load('dice_bau', lambda: DiceBase(mode=MODE_OPT).set_bau().run())
        dice_setup = {'ctrltype': {0: Ctrl.TYPE_EXTERNAL},
                              #'S': self._bau.S_year.loc[:self._time_dice.end].values,
                              'fex0': -self._bau.FORC_year.loc[2015]+self._bau.forcoth_year.loc[2015]+
                                             self._hist.forcing_year.loc[2015]}
        try:
            setup = dice_kwargs.pop('setup')
        except:
            setup = {}
        dice_setup.update(setup)
        try:
            dice_sow_setup = dice_kwargs.pop('sow_setup')
            dice_sow_setup['2015 forcings of non-CO2 GHG (Wm-2)'] = dice_sow_setup.get('Climate sensitivity (K)', 0)
        except:
            dice_sow_setup = {}
        simdice = DiceBase(endyear=endyear, mode=mode, setup=dice_setup, sow_setup=dice_sow_setup, **dice_kwargs)
        simdice.setfixed(simdice.TATM, 1, False)
        simdice.set_inbound('MIU', 'TATM').set_outbound('FORC')

        adapter = DiceDoeclimAdapter(simdice, simdoeclim)

        controller = controlclass(adapter, **control_kwargs)

        super().__init__(controller, simdice, adapter, simdoeclim)

    def run(self, *args, **kwargs):
        controller, simdice, adapter, _ = self._mlist
        return super().run(*args, times=[controller.time, simdice.time, adapter.time, self._time_doeclim_run], **kwargs)


