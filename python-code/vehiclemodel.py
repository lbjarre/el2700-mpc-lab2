import numpy as np
import yaml

class CarModel:

    def __init__(self, params):

        self.z = np.array([0, 0, 0, 0])
        self._beta = 0

        self.update_functions = lambda z, u: np.array([
            z[0] + self.Ts*z[2]*np.cos(z[3] + u[1]),
            z[1] + self.Ts*z[2]*np.sin(z[3] + u[1]),
            z[2] + self.Ts*u[0],
            z[3] + self.Ts*z[3]*np.sin(u[1])/params['l_r']
        ])

        self.a_max = params['a_max']
        self.beta_max = np.radians(params['beta_max'])
        self.beta_dot_max = np.radians(params['beta_dot_max'])
        self.Ts = params['Ts']

    def update_state(self, u):
        # check inputs for constrain violations
        a = min(abs(u[0]), self.a_max)
        beta = min(abs(u[1]), self.beta_max)
        beta_dot = (beta - self._beta)/self.Ts
        if abs(beta_dot) > self.beta_dot_max:
            beta = self._beta + np.sign(beta_dot)*self.Ts*self.beta_dot_max

        # update the states
        self.z = self.update_functions(self.z, np.array([a, beta]))
        self._beta = beta
        return self.z

    def run_sim(self, controller):
        # create time vector, allocate state and input vectors
        t_vec = np.arange(0, 10, self.Ts)
        z_vec = np.zeros((len(t_vec), 4))
        u_vec = np.zeros((len(t_vec), 2))
        z_vec[0, :] = self.z

        # simulate each time step
        for i, t in enumerate(t_vec):
            u_vec[i, :] = controller.calc_control(self.z)
            z_vec[i, :] = self.update_state(u_vec[i, :])
            print('Time {0} solved'.format(t))

        # reset to initial state for future simulations
        self.z = np.array([0, 0, 0, 0])

        return t_vec, z_vec, u_vec
