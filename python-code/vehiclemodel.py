import numpy as np
import yaml

class CarModel:

    def __init__(self, params):

        self.z = np.array([0, 0, 0, 0])
        self._beta = 0

        self.update_functions = lambda a, beta: np.array([
            self.z[2]*np.cos(self.z[3] + beta),
            self.z[2]*np.sin(self.z[3] + beta),
            a,
            self.z[3]*np.sin(beta)/params['l_r']
        ])

        self.a_max = params['a_max']
        self.beta_max = np.radians(params['beta_max'])
        self.beta_dot_max = np.radians(params['beta_dot_max'])
        self.Ts = params['Ts']

    def update_state(self, u):

        # check inputs for constrain violations
        a = min(abs(u[0]), self.a_max)
        beta = min(abs(u[1]), self.beta_max)
        if abs((u[1] - self._beta)/self.Ts) > self.beta_dot_max:
            beta = self._beta + self.Ts*self.beta_dot_max

        # update the states
        self.z = self.z + self.Ts*self.update_functions(a, beta)
        self._beta = beta
        return self.z
