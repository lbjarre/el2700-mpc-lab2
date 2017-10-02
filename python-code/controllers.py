import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import time

class MPC:

    def __init__(self, Q1, Q2, Qf, N):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Qf = Qf
        self.N = N
        self.constraints = (
            {
                'type': 'eq',
                'fun': self.equality_constraints
            },
            {
                'type': 'ineq',
                'fun': self.inequality_constraints
            }
        )

    def initialize(self, track, car_model):
        self.beta0 = car_model._beta
        self._x_opt = None
        self.Ts = car_model.Ts
        self.model_dynamics = car_model.get_model_dynamics(self.Ts)
        z_bound = (
            (-np.inf, np.inf),
            (track.y_edge_lo, track.y_edge_hi),
            (-np.inf, np.inf),
            (-np.inf, np.inf)
        )
        u_bound = (
            (-car_model.a_max, car_model.a_max),
            (-car_model.beta_max, car_model.beta_max)
        )
        self.bounds = np.vstack((
            np.tile(z_bound, (self.N - 1, 1)),
            np.tile(u_bound, (self.N - 1, 1))
        ))
        self.beta_dot_max = car_model.beta_dot_max
        self.x_goal = track.x_end
        self.obstacles = track.obstacles

    def xTQx(self, x, Q):
        return np.dot(np.dot(np.transpose(x), Q), x)

    def get_ref_err(self, x, ref, k):
        '''Calculates the reference error \hat{r} - r_{ref}'''
        pos = self.get_state(x, k)[0:2]
        len_ref = ref.shape[0]
        if k >= len_ref:
            pos_ref = ref[-1, :]
        else:
            pos_ref = ref[k, :]
        return pos - pos_ref

    def get_control(self, x, i):
        '''Gets z(i) from the optimization variable x'''
        start = 4*(self.N - 1) + 2*i
        return x[start: start + 2]

    def get_state(self, x, i):
        '''Gets u(i) from the optimization variable x'''
        if i == 0:
            return self.z0
        start = 4*(i-1)
        return x[start: start + 4]

    def set_state(self, x, z, i):
        start = 4*(i-1)
        x[start: start + 4] = z
        return x

    def set_control(self, x, u, i):
        start = 4*(self.N - 1) + 2*i
        x[start: start + 2] = u
        return x

    def get_model_constraints(self, x):
        f_eq = np.zeros(4*(self.N - 1))
        z_prev = self.z0
        for k in range(self.N - 1):
            curr_u = self.get_control(x, k)
            new_state = self.model_dynamics(z_prev, curr_u)
            f_eq[4*k: 4*(k+1)] = new_state - self.get_state(x, k+1)
            z_prev = new_state
        return f_eq

    def get_input_constraints(self, x):
        f_ineq = np.zeros(self.N - 1)
        beta_prev = self.beta0
        for k in range(self.N - 1):
            curr_u = self.get_control(x, k)
            f_ineq[k] = self.beta_dot_max - abs((curr_u[1] - beta_prev)/self.Ts)
            beta_prev = curr_u[1]
        return f_ineq

    def calc_control(self, z, partial_tracking=False):
        '''Calculate the MPC control given the current state z and the previous
           wheel angle beta. Solves the optimization problem and returns the
           first input step and the evaluated cost.'''
        self.z0 = z
        t_start = time.process_time()
        init_guess = self.calc_init_guess()
        res = opt.minimize(self.cost_function,
                           init_guess,
                           method='SLSQP',
                           constraints=self.constraints,
                           bounds=self.bounds,
                           options={
                               'maxiter': 250,
                               'disp': True,
                               #'ftol': 0.01
                           })
        t_solve = time.process_time() - t_start
        if partial_tracking:
            self.print_partial_progress(res.x, init_guess)
        self._x_opt = res.x
        u_opt = self.get_control(res.x, 0)
        self.beta0 = u_opt[1]
        return u_opt, res.fun, res.nit, t_solve

    def calc_init_guess(self):
        if self._x_opt is None:
            return self.generate_new_guess()
        x_init = np.zeros(6*(self.N - 1))
        for k in range(self.N - 2):
            _z_opt = self.get_state(self._x_opt, k + 2)
            x_init = self.set_state(x_init, _z_opt, k + 1)
            _u_opt = self.get_control(self._x_opt, k + 1)
            x_init = self.set_control(x_init, _u_opt, k)
        z_N = self.model_dynamics(_z_opt, _u_opt)
        x_init = self.set_control(x_init, _u_opt, self.N - 2)
        x_init = self.set_state(x_init, z_N, self.N - 1)
        for obs in self.obstacles:
            if obs.collision(z_N[0], z_N[1]):
                sgn = np.sign(obs.get_closest_edge_angle(_z_opt[0], _z_opt[1]))
                for n in range(self.N - 1, 0, -1):
                    z = self.get_state(x_init, n - 1)
                    u = self.get_control(x_init, n - 1)
                    _u = self.get_control(x_init, n - 2)
                    u[1] = _u[1] + sgn*self.Ts*self.beta_dot_max
                    for k in range(self.N - n):
                        z = self.model_dynamics(z, u)
                        x_init = self.set_control(x_init, u, n + k - 1)
                        x_init = self.set_state(x_init, z, n + k)
                        u[1] = u[1] + sgn*self.Ts*self.beta_dot_max
                    if not obs.collision(z[0], z[1]):
                        break
                else:
                    raise ValueError('cannot find init guess!')
        return x_init

    def generate_new_guess(self):
        x_init = np.zeros(6*(self.N - 1))
        for k in range(self.N - 1):
            new_state = self.model_dynamics(
                self.get_state(x_init, k),
                self.get_control(x_init, k))
            x_init = self.set_state(x_init, new_state, k + 1)
            for obs in self.obstacles:
                if obs.collision(new_state[0], new_state[1]):
                    _z = self.z0
                    _beta = self.beta0
                    sel = 1
                    for n in range(self.N - 1):
                        if sel == 1:
                            angle = obs.get_closest_edge_angle(_z[0], _z[1])
                            beta_desired = angle - _z[3]
                            beta_dot_desired = (beta_desired - _beta)/self.Ts
                            sgn = np.sign(beta_dot_desired)
                            sel = np.argmin(
                                [abs(beta_dot_desired), self.beta_dot_max]
                            )
                            _beta = [
                                beta_desired,
                                _beta + sgn*self.Ts*self.beta_dot_max
                            ][sel]
                        x_init = self.set_control(x_init, [0, _beta], n)
                        _z = self.model_dynamics(_z, [0, _beta])
                        x_init = self.set_state(x_init, _z, n + 1)
        return x_init

    def print_partial_progress(self, x_opt, x_init):
        z_opt = np.zeros((self.N, 4))
        z_init = np.zeros((self.N, 4))
        for k in range(self.N):
            z_opt[k, :] = self.get_state(x_opt, k)
            z_init[k, :] = self.get_state(x_init, k)
        fig = plt.figure()
        ax_xy = fig.add_subplot(1, 1, 1)
        [ax_xy.add_patch(
            ptch.Rectangle(
                *obs.get_plot_params(),
                facecolor='red',
                hatch='/'
            )
        ) for obs in self.obstacles]
        ax_xy.plot(z_init[:, 0], z_init[:, 1], color='b', linestyle='--', marker='+')
        ax_xy.plot(z_opt[:, 0], z_opt[:, 1], color='g', marker='+')
        ax_xy.set_title('XY-plot')
        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_xlim([0, 50])
        ax_xy.set_ylim([-8, 8])
        plt.show()

class RefTrackMPC(MPC):

    def cost_function(self, x):
        J = 0
        for k in range(self.N-1):
            ref_err = self.get_ref_err(x, self.reference, k)
            ref_cost = self.xTQx(ref_err, self.Q1)
            curr_u = self.get_control(x, k)
            u_cost = self.xTQx(curr_u, self.Q2)
            J = J + ref_cost + u_cost
        ref_err = self.get_ref_err(x, self.reference, k)
        term_cost = self.xTQx(ref_err, self.Qf)
        return J + term_cost

    def equality_constraints(self, x):
        return self.get_model_constraints(x)

    def inequality_constraints(self, x):
        return self.get_input_constraints(x)

class ObstacleAvoidMPC(MPC):

    def cost_function(self, x):
        J = 0
        for k in range(self.N-1):
            curr_u = self.get_control(x, k)
            curr_z = self.get_state(x, k)
            u_cost = self.xTQx(curr_u, self.Q2)
            dist_to_goal = self.x_goal - curr_z[0]
            J = J + u_cost + self.Qf*dist_to_goal
        term_z = self.get_state(x, self.N)
        dist_to_goal = self.x_goal - term_z[0]
        return J + dist_to_goal*self.Qf

    def equality_constraints(self, x):
        return self.get_model_constraints(x)

    def inequality_constraints(self, x):
        f_input = self.get_input_constraints(x)
        n_obs = len(self.obstacles)
        f_obs = np.zeros(n_obs*(self.N - 1))
        for k in range(1, self.N):
            curr_z = self.get_state(x, k)
            for i, obs in enumerate(self.obstacles):
                f_obs[n_obs*(k-1)+i] = obs.closest_distance(curr_z[0], curr_z[1])
        return np.concatenate((f_input, f_obs))
