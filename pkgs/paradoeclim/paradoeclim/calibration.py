from paradigm import MODE_SIM, CALIBPARAM
from paradoeclim import DoeclimCalib

vin = ['t2co', 'kappa', 'alpha', 'dq10', 'dq20', 'delql0', 'delqo0', 'dpast20', 'dteaux10', 'dteaux20']


class DoeclimCalibLikelihood(DoeclimCalib):

    def __init__(self):
        super().__init__(mode=MODE_SIM,
                         vin=['t2co', 'kappa', 'alpha',
                              'temp_wsigma', 'heat_wsigma',
                              'temp_rho', 'heat_rho'])

    def _body(self):
        super()._body()

        self.temp_wsigma = self.new(CALIBPARAM, default=0.1, bounds=(0.05, 0.5))
        self.temp_wsigma = self.new(CALIBPARAM, default=2.1, bounds=(0.1, 10))
        self.temp_rho = self.new(CALIBPARAM, default=0.5, bounds=(0., 1-1e-2))
        self.heat_rho = self.new(CALIBPARAM, default=0.9, bounds=(0., 1-1e-2))


dc = DoeclimCalib(name='doeclim_sim', mode=MODE_SIM, vin=vin)
