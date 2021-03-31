
from paradigm import Model, VAR, OBJECTIVE, MODE_OPT
from pyomo.core import minimize
import pyomo.core.base.expr as ex
ex


class Simple(Model):

    def _body(self):
        self.x = self.new(VAR, doc='x', bounds=(-10,10), initialize=5)

    def _body_eqs(self):
        self.objective = self.new(OBJECTIVE, rule=lambda m: m.x*m.x, sense=minimize)


s = Simple(time=None, mode=MODE_OPT)
i = s.create_instance()
print(hex(id(i)), hex(id(i.parent_component())))