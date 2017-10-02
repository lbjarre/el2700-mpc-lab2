import numpy as np
import time

class CarModel(object):

    def __init__(self, params):
        self.z = np.array([0, 0, 10, 0])
        self._beta = 0
        self.l_r = params['l_r']
        self.a_max = params['a_max']
        self.beta_max = np.radians(params['beta_max'])
        self.beta_dot_max = np.radians(params['beta_dot_max'])
        self.Ts = params['Ts']

    def check_input_constrains(self, u):
        a = np.sign(u[0])*min(abs(u[0]), self.a_max)
        beta = np.sign(u[1])*min(abs(u[1]), self.beta_max)
        beta_dot = (beta - self._beta)/self.Ts
        if abs(beta_dot) > self.beta_dot_max:
            beta = self._beta + np.sign(beta_dot)*self.Ts*self.beta_dot_max
        return np.array([a, beta])

class NonlinearCarModel(CarModel):

    def model_dynamics(self, z, u, Ts):
        return z + Ts*np.array([
            z[2]*np.cos(z[3] + u[1]),
            z[2]*np.sin(z[3] + u[1]),
            u[0],
            z[3]*np.sin(u[1])/self.l_r
        ])

    def get_model_dynamics(self, Ts):
        return lambda z, u: self.model_dynamics(z, u, Ts)

    def update_state(self, u):
        u = self.check_input_constrains(u)
        self.z = self.model_dynamics(self.z, u, self.Ts)
        self._beta = u[1]
        return self.z

class LinearizedCarModel(CarModel):

    def __init__(self, params):
        super().__init__(params)
        self.v0 = self.z[2]
        self.A = np.array([
            [1, 0, self.Ts, 0],
            [0, 1, 0, self.v0*self.Ts],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            0,
            self.v0*self.Ts*(1 + (self.v0*self.Ts)/(2*self.l_r)),
            0,
            self.v0*self.Ts/self.l_r
        ])

    def model_dynamics(self, z, u):
        return np.dot(self.A, z) + np.dot(self.B, u)

    def get_model_dynamics(self, Ts):
        A = np.array([
            [1, 0, Ts, 0],
            [0, 1, 0, self.v0*Ts],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([
            0,
            self.v0*Ts*(1 + (self.v0*Ts)/(2*self.l_r)),
            0,
            self.v0*Ts/self.l_r
        ])
        return lambda z, u: np.dot(A, z) + np.dot(B, u)

    def update_state(self, u):
        u = self.check_input_constrains(u)
        self.z = self.model_dynamics(self.z, u)
        self._beta = u[1]
        return self.z
