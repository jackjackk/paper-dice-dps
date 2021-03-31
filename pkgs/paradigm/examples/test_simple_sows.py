from paradigm import Model, VAR, OBJECTIVE, minimize, MODE_SIM, RANGESET, PARAM
import numpy as np

class Simple(Model):

    def _body(self):
        self.i = self.new(RANGESET, 1, 3)
        self.a = self.new(PARAM, doc='a')
        self.x = self.new(VAR, self.i, doc='x', bounds=lambda m, i: (0*m.a, m.a), default=5)

    def _body_eqs(self):
        self.objective = self.new(OBJECTIVE, rule=lambda m: m.x*m.x, sense=minimize)



s = Simple(mode=MODE_SIM, default_sow=10, setup={'a': np.arange(10)})

s.x_bounds
