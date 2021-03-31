import logging
from math import pi, sqrt, erf
import pkgutil
import io

from pyomo.core.base import minimize, Constraint

from paradigm.model import SET, RANGESET, PARAM, VAR, EQUATION, CONSTRAINT, OBJECTIVE, MODE_SIM, CALIBPARAM
from paradigm.model import Model, Time, ex

import pandas as pd

from paradigm.model import ConfigurableModel, Equation

from .utils import get_hist_forc_data, get_hist_temp_data, get_hist_heat_data
import numpy as np

logger = logging.getLogger('doeclim')

def tfirst0fixer(varname, initvalue=None):
    def tfirst0fix(m, *indices):
        v = getattr(m, varname)
        if initvalue is None:
            p0 = getattr(m, varname.lower()+'0')
        else:
            p0 = initvalue
        if len(indices) == 0:
            m.setfixed(v, None)
        elif min(indices) == 1:
            m.setfixed(v, indices)
        return p0
    return tfirst0fix

forc2comp = {
    'forcing_nonaero': ['ghg', 'o3', 'sh2o', 'stra', 'solar', 'land'],
    'forcing_aero': ['refa', 'aie', 'bc', 'snow']
}

def forcing_hindcast_fixer(v, forc_data):
    def forc_fix(m, t):
        try:
            ret = forc_data.loc[m.year[t]]
            m.setfixed(v, t)
        except:
            fidx = forc_data.index.get_loc(m.year[t], method='nearest')
            ret = forc_data.iloc[fidx]
        return ret
    return forc_fix



class Doeclim(Model):

    state0 = ['DQ1', 'DQ2', 'QC1', 'QC2', 'QL', 'QO', 'DelQL', 'DelQO',
              'DPAST2', 'DTEAUX1', 'DTEAUX2']

    def _init_scalars(self):
        # self.forc_data = get_hist_forc_data()
        pass

    def _init_sets(self):
        # Sets
        self.t2 = self.new(RANGESET, 1, self.ns)


    def _init_params(self):

        # Calibrated parameters
        self.t2co = self.new(CALIBPARAM, doc='Climate sensitivity (K)', default=3.1, bounds=(2,4))
        self.kappa = self.new(CALIBPARAM, doc='Ocean heat diffusivity (cm^2 s^-1)', default=5.4, bounds=(1e-3, 1e2))
        self.alpha = self.new(CALIBPARAM, doc='Aerosol forcing scaling factor', default=0.5, bounds=(-10,10))

        self.temp0 = self.new(CALIBPARAM, doc='Initial temperature (K)', default=0., bounds=(-0.3,0.3))
        self.heat0 = self.new(CALIBPARAM, doc='Initial ocean heat (10^22 J)', default=0., bounds=(-50,0))

        self.forcing_source = self.new(PARAM, doc='forcing data source, 0 = GISS, 1 = Urban', default=0, sow=0)

        # Define the doeclim parameters
        self.ak = self.new(PARAM, default=0.31, sow=0)
        self.bk = self.new(PARAM, default=1.59, sow=0)
        self.csw = self.new(PARAM, default=0.13, sow=0)
        self.earth_area = self.new(PARAM, default=5100656 * pow(10.0, 8), sow=0)
        self.kcon = self.new(PARAM, default=3155.0, sow=0)
        self.q2co = self.new(PARAM, default=3.7, sow=0)
        self.rlam = self.new(PARAM, default=1.43, sow=0)
        self.secs_per_Year = self.new(PARAM, default=31556926.0, sow=0)
        self.zbot = self.new(PARAM, default=4000.0, sow=0)
        self.bsi = self.new(PARAM, default=1.3, sow=0)
        self.cal = self.new(PARAM, default=0.52, sow=0)
        self.cas = self.new(PARAM, default=7.80, sow=0)
        self.flnd = self.new(PARAM, default=0.29, sow=0)
        self.fso = self.new(PARAM, default=0.95, sow=0)

        # Dependent Model Parameters
        self.ocean_area = self.new(PARAM, initialize=lambda m: (1.0 - m.flnd) * m.earth_area, sow=0)
        self.cnum = self.new(PARAM, initialize=lambda m: m.rlam * m.flnd + m.bsi * (1.0 - m.flnd), sow=0)
        self.cden = self.new(PARAM, initialize=lambda m: m.rlam * m.flnd - m.ak * (m.rlam - m.bsi), sow=0)
        self.cfl = self.new(CALIBPARAM, initialize=lambda m: m.flnd * m.cnum / m.cden * m.q2co / m.t2co - m.bk * (m.rlam - m.bsi) / m.cden)
        self.cfs = self.new(CALIBPARAM, initialize=lambda m: (m.rlam * m.flnd - m.ak / (1.0 - m.flnd) * (m.rlam - m.bsi)) * m.cnum / m.cden * m.q2co / m.t2co + m.rlam * m.flnd / (1.0 - m.flnd) * m.bk * (m.rlam - m.bsi) / m.cden)
        self.kls = self.new(CALIBPARAM, initialize=lambda m: m.bk * m.rlam * m.flnd / m.cden - m.ak * m.flnd * m.cnum / m.cden * m.q2co / m.t2co)
        self.keff = self.new(CALIBPARAM, initialize=lambda m: m.kcon * m.kappa)
        self.taubot = self.new(CALIBPARAM, initialize=lambda m: pow(m.zbot,2) / m.keff)
        self.powtoheat = self.new(PARAM, initialize=lambda m: m.ocean_area * m.secs_per_Year / pow(10.0,22))
        self.taucfs = self.new(CALIBPARAM, initialize=lambda m: m.cas / m.cfs)
        self.taucfl = self.new(CALIBPARAM, initialize=lambda m: m.cal / m.cfl)
        self.taudif = self.new(CALIBPARAM, initialize=lambda m: pow(m.cas,2) / pow(m.csw,2) * pi / m.keff)
        self.tauksl = self.new(CALIBPARAM, initialize=lambda m: (1.0 - m.flnd) * m.cas / m.kls)
        self.taukls = self.new(CALIBPARAM, initialize=lambda m: m.flnd * m.cal / m.kls)

        # First order
        self.KT0 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i:
                       4.0 - 2.0 * pow(2.0, 0.5) if (i-1) == m.ns-1 else
                       4.0 * pow(((m.ns-(i-1))), 0.5) - 2.0 * pow(((m.ns+1-(i-1))), 0.5) - 2.0 * pow((m.ns-1-(i-1)), 0.5), vectorize=False)
        self.KTA1 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        -8.0 * m.exp(-m.taubot / m.dt) + 4.0 * pow(2.0, 0.5) * m.exp(-0.5 * m.taubot / (m.dt)) if (i-1) == m.ns-1 else
                        -8.0 * pow((m.ns-(i-1)), 0.5) * m.exp(-m.taubot / (m.dt) / (m.ns-(i-1))) + 4.0 * pow((m.ns+1-(i-1)), 0.5) * m.exp(-m.taubot / (m.dt) / (m.ns+1-(i-1))) + 4.0 * pow((m.ns-1-(i-1)), 0.5) * m.exp(-m.taubot/(m.dt) / (m.ns-1-(i-1))), vectorize=False)
        self.KTB1 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        4.0 * pow((pi * m.taubot / (m.dt)), 0.5) * (1.0 + m.erf(pow(0.5 * m.taubot / (m.dt), 0.5)) - 2.0 * m.erf(pow(m.taubot / (m.dt), 0.5))) if (i-1) == m.ns-1 else
                        4.0 * pow((pi * m.taubot / (m.dt)), 0.5) * ( m.erf(pow((m.taubot / (m.dt) / (m.ns-1-(i-1))), 0.5)) + m.erf(pow((m.taubot / (m.dt) / (m.ns+1-(i-1))), 0.5)) - 2.0 * m.erf(pow((m.taubot / (m.dt) / (m.ns-(i-1))), 0.5)) ), vectorize=False)
        # Second order
        self.KTA2 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        8.0 * m.exp(-4.0 * m.taubot / (m.dt)) - 4.0 * pow(2.0, 0.5) * m.exp(-2.0 * m.taubot / (m.dt)) if (i-1) == m.ns-1 else
                        8.0 * pow((m.ns-(i-1)), 0.5) * m.exp(-4.0 * m.taubot / (m.dt) / (m.ns-(i-1))) - 4.0 * pow((m.ns+1-(i-1)), 0.5) * m.exp(-4.0 * m.taubot / (m.dt) / (m.ns+1-(i-1))) - 4.0 * pow((m.ns-1-(i-1)), 0.5) * m.exp(-4.0 * m.taubot / (m.dt) / (m.ns-1-(i-1))), vectorize=False)
        self.KTB2 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        -8.0 * pow((pi * m.taubot / (m.dt)), 0.5) * (1.0 + m.erf(pow((2.0 * m.taubot / (m.dt)), 0.5)) - 2.0 * m.erf(2.0 * pow((m.taubot / (m.dt)), 0.5)) ) if (i-1) == m.ns-1 else
                        -8.0 * pow((pi * m.taubot / (m.dt)), 0.5) * ( m.erf(2.0 * pow((m.taubot / (m.dt) / (m.ns-1-(i-1))), 0.5)) + m.erf(2.0 * pow((m.taubot / (m.dt) / (m.ns+1-(i-1))), 0.5)) - 2.0 * m.erf(2.0 * pow((m.taubot / (m.dt) / (m.ns-(i-1))), 0.5)) ), vectorize=False)
        # Third order
        self.KTA3 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        -8.0 * m.exp(-9.0 * m.taubot / (m.dt)) + 4.0 * pow(2.0, 0.5) * m.exp(-4.5 * m.taubot / (m.dt)) if (i-1) == m.ns-1 else
                        -8.0 * pow((m.ns-(i-1)), 0.5) * m.exp(-9.0 * m.taubot / (m.dt) / (m.ns-(i-1))) + 4.0 * pow((m.ns+1-(i-1)), 0.5) * m.exp(-9.0 * m.taubot / (m.dt) / (m.ns+1-(i-1))) + 4.0 * pow((m.ns-1-(i-1)), 0.5) * m.exp(-9.0 * m.taubot / (m.dt) / (m.ns-1-(i-1))), vectorize=False)
        self.KTB3 = self.new(CALIBPARAM, self.t2, initialize=lambda m,i :
                        12.0 * pow((pi * m.taubot / (m.dt)), 0.5) * (1.0 + m.erf(pow((4.5 * m.taubot / (m.dt)), 0.5)) - 2.0 * m.erf(3.0 * pow((m.taubot / (m.dt)), 0.5)) ) if (i-1) == m.ns-1 else
                        12.0 * pow((pi * m.taubot / (m.dt)), 0.5) * ( m.erf(3.0 * pow((m.taubot / (m.dt) / (m.ns-1-(i-1))), 0.5)) + m.erf(3.0 * pow((m.taubot / (m.dt) / (m.ns+1-(i-1))), 0.5)) - 2.0 * m.erf(3.0 * pow((m.taubot / (m.dt) / (m.ns-(i-1))), 0.5)) ), vectorize=False)
        # Sum up the kernel components
        self.Ker = self.new(CALIBPARAM, self.t2, initialize=lambda m,i : m.KT0[i] + m.KTA1[i] + m.KTB1[i] + m.KTA2[i] + m.KTB2[i] + m.KTA3[i] + m.KTB3[i], vectorize=False)
        # Switched on (To switch off, comment out lines below)
        self.C0 = self.new(CALIBPARAM, initialize=lambda m:   (1.0 / pow(m.taucfl, 2.0) + 1.0 / pow(m.taukls, 2.0) + 2.0 / m.taucfl / m.taukls + m.bsi / m.taukls / m.tauksl)*(pow((m.dt), 2.0) / 12.0))
        self.C1 = self.new(CALIBPARAM, initialize=lambda m:   (-1 * m.bsi / pow(m.taukls, 2.0) - m.bsi / m.taucfl / m.taukls - m.bsi / m.taucfs / m.taukls - pow(m.bsi, 2.0) / m.taukls / m.tauksl)*(pow((m.dt), 2.0) / 12.0))
        self.C2 = self.new(CALIBPARAM, initialize=lambda m:   (-1 * m.bsi / pow(m.tauksl, 2.0) - 1.0 / m.taucfs / m.tauksl - 1.0 / m.taucfl / m.tauksl -1.0 / m.taukls / m.tauksl)*(pow((m.dt), 2.0) / 12.0))
        self.C3 = self.new(CALIBPARAM, initialize=lambda m:   (1.0 / pow(m.taucfs, 2.0) + pow(m.bsi, 2.0) / pow(m.tauksl, 2.0) + 2.0 * m.bsi / m.taucfs / m.tauksl + m.bsi / m.taukls / m.tauksl)*(pow((m.dt), 2.0) / 12.0))

        #------------------------------------------------------------------
        # Matrices of difference equation system B*T(i+1) = Q(i) + A*T(i)
        # T = (TL,TO)
        # (Equation A.27, EK05, or Equations 2.3.24 and 2.3.27, TK07)
        self.B0 = self.new(CALIBPARAM, initialize=lambda m:  1.0 + (m.dt) / (2.0 * m.taucfl) + (m.dt) / (2.0 * m.taukls) + m.C0)
        self.B1 = self.new(CALIBPARAM, initialize=lambda m:  (-m.dt) / (2.0 * m.taukls) * m.bsi + m.C1)
        self.B2 = self.new(CALIBPARAM, initialize=lambda m:  (-m.dt) / (2.0 * m.tauksl) + m.C2)
        self.B3 = self.new(CALIBPARAM, initialize=lambda m:  1.0 + (m.dt) / (2.0 * m.taucfs) + (m.dt) / (2.0 * m.tauksl) * m.bsi + 2.0 * m.fso * pow(((m.dt) / m.taudif), 0.5) + m.C3)

        self.A0 = self.new(CALIBPARAM, initialize=lambda m:  1.0 - (m.dt) / (2.0 * m.taucfl) - (m.dt) / (2.0 * m.taukls) + m.C0)
        self.A1 = self.new(CALIBPARAM, initialize=lambda m:  (m.dt) / (2.0 * m.taukls) * m.bsi + m.C1)
        self.A2 = self.new(CALIBPARAM, initialize=lambda m:  (m.dt) / (2.0 * m.tauksl) + m.C2)
        self.A3 = self.new(CALIBPARAM, initialize=lambda m:  1.0 - (m.dt) / (2.0 * m.taucfs) - (m.dt) / (2.0 * m.tauksl) * m.bsi + m.Ker[m.ns] * m.fso * pow(((m.dt) / m.taudif), 0.5) + m.C3)

        # Calculate the inverse of B
        self.IB0 = self.new(CALIBPARAM, initialize=lambda m: 1/(m.B0*m.B3 - m.B1*m.B2) * m.B3)
        self.IB1 = self.new(CALIBPARAM, initialize=lambda m: 1/(m.B0*m.B3 - m.B1*m.B2) * -1 * m.B1)
        self.IB2 = self.new(CALIBPARAM, initialize=lambda m: 1/(m.B0*m.B3 - m.B1*m.B2) * -1 * m.B2)
        self.IB3 = self.new(CALIBPARAM, initialize=lambda m: 1/(m.B0*m.B3 - m.B1*m.B2) * m.B0)
        #IB = np.ravel(np.linalg.inv(np.reshape(B, [2,2])))
        
        # Initial state
        self.dq10 = self.new(PARAM, default=0)
        self.dq20 = self.new(PARAM, default=0)
        self.delql0 = self.new(PARAM, default=0)
        self.delqo0 = self.new(PARAM, default=0)
        self.dpast20 = self.new(PARAM, default=0)
        self.dteaux10 = self.new(PARAM, default=0)
        self.dteaux20 = self.new(PARAM, default=0)


    def _init_variables(self):
        # Initialize variables for time-stepping through the model
        self.DQ1 = self.new(VAR, self.t, initialize=tfirst0fixer('DQ1'))
        self.DQ2 = self.new(VAR, self.t, initialize=tfirst0fixer('DQ2'))
        self.QC1 = self.new(VAR, self.t)
        self.QC2 = self.new(VAR, self.t)
        self.QL = self.new(VAR, self.t)
        self.QO = self.new(VAR, self.t)
        self.DelQL = self.new(VAR, self.t, initialize=tfirst0fixer('DelQL'))
        self.DelQO = self.new(VAR, self.t, initialize=tfirst0fixer('DelQO'))
        self.DPAST1 = self.new(VAR, initialize=tfirst0fixer('DPAST1', 0))
        self.DPAST2 = self.new(VAR, self.t, initialize=tfirst0fixer('DPAST2'))
        #self.DPAST2b = self.new(VAR, self.t, self.t2, initialize=tfirst0fixer('DPAST2b'))
        self.DTEAUX1 = self.new(VAR, self.t, initialize=tfirst0fixer('DTEAUX1'))
        self.DTEAUX2 = self.new(VAR, self.t, initialize=tfirst0fixer('DTEAUX2'))

        # Reset the endogenous varibales for this time step
        self.temp = self.new(VAR, self.t, doc='Global mean temperature anomaly (K), preindustrial')
        self.temp_landair = self.new(VAR, self.t, doc='Land air temperature anomaly (K)')
        self.temp_sst = self.new(VAR, self.t, doc='Sea surface temperature anomaly (K)')
        self.heat = self.new(VAR, self.t, doc='Mixed + Interior heat anomaly (10^22 J)')
        self.heat_mixed = self.new(VAR, self.t, doc='Mixed layer heat anomaly (10^22 J)', initialize=tfirst0fixer('heat_mixed', 0))
        self.heat_interior = self.new(VAR, self.t, doc='Interior ocean heat anomaly (10^22 J)', initialize=tfirst0fixer('heat_interior', 0))
        self.heatflux_mixed = self.new(VAR, self.t, doc='Heat uptake of the mixed layer (W/m^2)', initialize=tfirst0fixer('heatflux_mixed', 0))
        self.heatflux_interior = self.new(VAR, self.t, doc='Heat uptake of the interior ocean (W/m^2)', initialize=tfirst0fixer('heatflux_interior', 0))
        #self.heatflux_interior1 = self.new(VAR, self.t, self.t, doc='Aux var for calculating Heat uptake of the interior ocean (W/m^2)', initialize=tfirst0fixer('heatflux_interior1'))
        self.forcing = self.new(VAR, self.t, doc='Radiative forcing (W/m2)')
        forc_data = get_hist_forc_data()
        forc_source2label = ['giss', 'urban']
        forc_data4fix = lambda v: forc_data.loc[forc_source2label[int(self.forcing_source.value)]][v]
        self.forcing_nonaero = self.new(VAR, self.t, sow=0, doc='Radiative forcing - excluding aerosols (W/m2)', initialize=forcing_hindcast_fixer('forcing_nonaero', forc_data4fix('non-aerosols')))
        self.forcing_aero = self.new(VAR, self.t, sow=0, doc='Radiative forcing - excluding aerosols (W/m2)', initialize=forcing_hindcast_fixer('forcing_aero', forc_data4fix('aerosols')))


    def _init_equations(self):
        # Forcing
        self.forcingeq = self.new(EQUATION, self.forcing, lambda m,tstep: m.forcing_nonaero[tstep] + m.alpha*m.forcing_aero[tstep])

        # Assume land and ocean forcings are equal to global forcing
        self.QLeq = self.new(EQUATION, self.QL, lambda m,tstep: m.forcing[tstep])
        self.QOeq = self.new(EQUATION, self.QO, lambda m,tstep: m.forcing[tstep])

        self.DelQLeq = self.new(EQUATION, self.DelQL, lambda m,tstep: m.QL[tstep] - m.QL[tstep - 1] if tstep>1 else None)
        self.DelQOeq = self.new(EQUATION, self.DelQO, lambda m,tstep: m.QO[tstep] - m.QO[tstep - 1] if tstep>1 else None)

        # Assume linear forcing change between tstep and tstep+1
        self.QC1eq = self.new(EQUATION, self.QC1, lambda m,tstep: (m.DelQL[tstep]/m.cal*(1.0/m.taucfl+1.0/m.taukls)-m.bsi*m.DelQO[tstep]/m.cas/m.taukls) * pow((m.dt), 2.0)/12.0)
        self.QC2eq = self.new(EQUATION, self.QC2, lambda m,tstep: (m.DelQO[tstep]/m.cas*(1.0/m.taucfs+m.bsi/m.tauksl)-m.DelQL[tstep]/m.cal/m.tauksl) * pow((m.dt), 2.0)/12.0)

        # ----------------- Initial Conditions --------------------
        # Initialization of temperature and forcing vector:
        # Factor 1/2 in front of Q in Equation A.27, EK05, and Equation 2.3.27, TK07 is a typo!
        # Assumption: linear forcing change between n and n+1
        self.DQ1eq = self.new(EQUATION, self.DQ1, lambda m,tstep: 0.5*(m.dt)/m.cal*(m.QL[tstep]+m.QL[tstep-1]) + m.QC1[tstep] if tstep>1 else None)
        self.DQ2eq = self.new(EQUATION, self.DQ2, lambda m,tstep: 0.5*(m.dt)/m.cas*(m.QO[tstep]+m.QO[tstep-1]) + m.QC2[tstep] if tstep>1 else None)

        # ---------- SOLVE MODEL ------------------
        # Self.Calculate temperatures

        #self.DPAST2beq = self.new(EQUATION, self.DPAST2b, lambda m,tstep,i: m.DPAST2b[(tstep,i-1)] + m.temp_sst[i-1] * m.Ker[m.ns-tstep+i-1] if ((tstep>1) and (i>1) and (i<=tstep)) else None)
        #self.DPAST2eq = self.new(EQUATION, self.DPAST2, lambda m,tstep: m.fso * pow(((m.dt)/m.taudif), 0.5) * (m.DPAST2b[(tstep,tstep)] + m.temp_sst[tstep] * m.Ker[m.ns]) if tstep>1 else None)
        if self._use_numpy:
            self.DPAST2eq = self.new(EQUATION, self.DPAST2,lambda m, tstep:
                    #m.fso * pow(((m.dt) / m.taudif), 0.5) * sum(m.temp_sst[i] * m.Ker[m.ns-tstep+i] for i in range(1,tstep)) if tstep > 1 else None)
                m.fso * pow(((m.dt) / m.taudif), 0.5) * np.sum(np.multiply(m.temp_sst[1:tstep], m.Ker[(m.ns - tstep + 1):(m.ns)]), 0) if tstep > 1 else None)
        else:
            self.DPAST2eq = self.new(EQUATION, self.DPAST2,lambda m, tstep:
                    m.fso * pow(((m.dt) / m.taudif), 0.5) * sum(m.temp_sst[i] * m.Ker[m.ns-tstep+i] for i in range(1,tstep)) if tstep > 1 else None)

        self.DTEAUX1eq = self.new(EQUATION, self.DTEAUX1, lambda m,tstep: m.A0 * m.temp_landair[tstep-1] + m.A1 * m.temp_sst[tstep-1] if tstep>1 else None)
        self.DTEAUX2eq = self.new(EQUATION, self.DTEAUX2, lambda m,tstep: m.A2 * m.temp_landair[tstep-1] + m.A3 * m.temp_sst[tstep-1] if tstep>1 else None)

        self.temp_landaireq = self.new(EQUATION, self.temp_landair, lambda m,tstep: m.IB0 * (m.DQ1[tstep] + m.DPAST1 + m.DTEAUX1[tstep]) + m.IB1 * (m.DQ2[tstep] + m.DPAST2[tstep] + m.DTEAUX2[tstep]))
        self.temp_ssteq = self.new(EQUATION, self.temp_sst, lambda m,tstep: m.IB2 * (m.DQ1[tstep] + m.DPAST1 + m.DTEAUX1[tstep]) + m.IB3 * (m.DQ2[tstep] + m.DPAST2[tstep] + m.DTEAUX2[tstep]))

        self.tempeq = self.new(EQUATION, self.temp, lambda m,tstep: m.temp0 + m.flnd * m.temp_landair[tstep] + (1.0 - m.flnd) * m.bsi * m.temp_sst[tstep])

        if self._calculate_heat:
            # Calculate ocean heat uptake [W/m^2]
            # self.heatflux[tstep] captures in the heat flux in the period between tstep-1 and tstep.
            # Numerical implementation of Equation 2.7, EK05, or Equation 2.3.13, TK07)
            # ------------------------------------------------------------------------
            self.heatflux_mixedeq = self.new(EQUATION, self.heatflux_mixed, lambda m,tstep: m.cas*(m.temp_sst[tstep] - m.temp_sst[tstep-1]) if tstep>1 else None)
            self.heatflux_interioreq = self.new(EQUATION, self.heatflux_interior, lambda m, tstep:
                    m.cas * m.fso / pow((m.taudif * m.dt), 0.5) * (2.0 * m.temp_sst[tstep] - sum(m.temp_sst[i-1]*m.Ker[m.ns-tstep+i] for i in range(2,tstep+1))) if tstep > 1 else None)
            self.heat_mixedeq = self.new(EQUATION, self.heat_mixed, lambda m,tstep: m.heat_mixed[tstep-1] + m.heatflux_mixed[tstep] * (m.powtoheat*m.dt) if tstep>1 else None)
            self.heat_interioreq = self.new(EQUATION, self.heat_interior, lambda m,tstep: m.heat_interior[tstep-1] + m.heatflux_interior[tstep] * (m.fso*m.powtoheat*m.dt) if tstep>1 else None)
            self.heateq = self.new(EQUATION, self.heat, lambda m, tstep: m.heat0 + m.heat_interior[tstep] + m.heat_mixed[tstep])


    def __init__(self, calib=False, calculate_heat=False, use_numpy=True, **kwargs):
        if calib:
            _time = Time(start=1900,end=2015,tstep=1)
        else:
            _time = Time(start=1900,end=2100,tstep=1)
        time = kwargs.pop('time', _time)
        self._calculate_heat = calculate_heat
        self._use_numpy = use_numpy
        super().__init__(time, calib=calib, **kwargs)


#class DoeclimStandalone(Doeclim):
#    def __init__(self):
#        self.dclim = Doeclim()
#        self.obj = Objective(self.dclim.)


class DoeclimCalib(Doeclim):

    def _init_params(self):
        super()._init_params()
        self.temphist = self.new(PARAM, self.t) #, initialize=lambda m, i: m.tempdata.loc[min(m.year[i],m.tempdata.index[-1])], sow=0)
        self.set(temphist=get_hist_temp_data(self))
        self.heathist = self.new(PARAM, self.t) #, initialize=lambda m, i: m.tempdata.loc[min(m.year[i],m.tempdata.index[-1])], sow=0)
        self.set(heathist=get_hist_heat_data(self))


    def _init_variables(self):
        super()._init_variables()
        #self.temp_err = self.new(VAR, self.t)
        #self.heat_err = self.new(VAR, self.t)

    def _init_equations(self):
        super()._init_equations()
        #self.temp_erreq = self.new(EQUATION, self.temp_err, lambda m,i: m.temp[i] - m.temphist[i])
        #self.heat_erreq = self.new(EQUATION, self.heat_err, lambda m, i: m.heat[i] - m.heathist[i])

        #self.obj = self.new(OBJECTIVE, rule=lambda m: m.summation(m.err), sense=minimize)

    def __init__(self, **kwargs):
        super().__init__(calib=True, calculate_heat=True, **kwargs)

def doeclimCalibSim():
    return DoeclimCalib(name='doeclim_sim', setup={'ns':{None:116}}, mode=MODE_SIM, vin=['t2co','kappa','alpha'])


#dc = DoeclimCalib().set('ns',405)
#s = Simulator(dc, ['t2co','kappa','alpha'], simtime='tstep')
#s([3, 81.366475960442429, -4.4379154101656484], )
#s([3, 3.059999942779541, 3.7799999713897705])
#s([3, 3.059999942779541, 0.44]) #3.7799999713897705])
#s([3, 0.55, 0.44]) #3.7799999713897705])
"""
nvars, vbounds, nobjs, nconstrs, senses, f = model2simulator(DoeclimCalib().set('ns', 405), vin=['t2co','kappa','alpha'], vout=['obj'], simtime='tstep')
f(3., 3.3600000540415444, 3.7199999491373696)

#a = dc.set('t2co', 3).solve()
import matplotlib.pylab as plt
fig, ax = plt.subplots(1,1)
a.data['temp'].plot(ax=ax)
#a.insts[0].pprint()
a.insts[0].tempdata.loc[1900:].shift().reset_index()['J-D'].plot(ax=ax, label='data')
"""
