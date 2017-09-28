import numpy as np
import yaml

class CarModel:

    def __init__(self, params):

        self.z = np.array([0, 0, 10, 0])
        self._beta = 0

        self.update_functions = lambda z, u: z + self.Ts*np.array([
            z[2]*np.cos(z[3] + u[1]),
            z[2]*np.sin(z[3] + u[1]),
            u[0],
            z[3]*np.sin(u[1])/params['l_r']
        ])

        self.a_max = params['a_max']
        self.beta_max = np.radians(params['beta_max'])
        self.beta_dot_max = np.radians(params['beta_dot_max'])
        self.Ts = params['Ts']

    def update_state(self, u):
        # check inputs for constrain violations
        a = np.sign(u[0])*min(abs(u[0]), self.a_max)
        beta = np.sign(u[1])*min(abs(u[1]), self.beta_max)
        beta_dot = (beta - self._beta)/self.Ts
        if abs(beta_dot) > self.beta_dot_max:
            beta = self._beta + np.sign(beta_dot)*self.Ts*self.beta_dot_max

        # update the states
        self.z = self.update_functions(self.z, np.array([a, beta]))
        self._beta = beta
        return self.z

    def run_sim(self, controller, reference):
        # create time vector, allocate state and input vectors
        # t_vec = np.arange(0, 10, self.Ts)
        # z_vec = np.zeros((len(t_vec), 4))
        # u_vec = np.zeros((len(t_vec), 2))
        # j_vec = np.zeros(len(t_vec))
        # z_vec[0, :] = self.z
        #
        # # simulate each time step
        # for i, t in enumerate(t_vec):
        #     u_vec[i, :], j_vec[i] = controller.calc_control(self.z, self._beta, reference[i:, :])
        #     z_vec[i, :] = self.update_state(u_vec[i, :])
        #     print('Time {:1.1f} solved'.format(t))

        t_vec = np.array([None])
        z_vec = np.array([None, None, None, None])
        u_vec = np.array([None, None])
        j_vec = np.array([None])
        i = 0
        z = self.z
        while True:
            u, j = controller.calc_control(self.z, self._beta, reference[i:, :])
            t_vec = np.vstack((t_vec, i*self.Ts))
            z_vec = np.vstack((z_vec, z))
            u_vec = np.vstack((u_vec, u))
            j_vec = np.vstack((j_vec, j))
            print('Time {:1.1f} solved'.format(i*self.Ts))
            if z[0] >= controller.x_goal:
                break
            z = self.update_state(u)
            i = i + 1

        # reset to initial state for future simulations
        self.z = np.array([0, 0, 10, 0])

        return t_vec, z_vec, u_vec, j_vec
