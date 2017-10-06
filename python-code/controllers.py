import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import time

class MPC:

    def __init__(self, Qz, Qu, Qf, N, margin):
        self.nz = 4
        self.nu = 2
        self.Qz = Qz
        self.Qu = Qu
        self.Qf = Qf
        self.N = N
        self.margin = margin
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
        '''Initialize the controller for the track model and the car model.'''
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
        u_bound_2 = (
            (-car_model.a_max, car_model.a_max),
            (-car_model.beta_max, car_model.beta_max)
        )
        u_bound_1 = (
            (-car_model.beta_max, car_model.beta_max)
        )
        if self.nu == 1:
            u_bound = u_bound_1
        else:
            u_bound = u_bound_2
        self.bounds = np.vstack((
            np.tile(z_bound, (self.N - 1, 1)),
            np.tile(u_bound, (self.N - 1, 1))
        ))
        self.beta_dot_max = car_model.beta_dot_max
        self.x_goal = track.x_end
        self.obstacles = track.obstacles

    def xTQx(self, x, Q):
        '''Returns the value of x^TQx'''
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
        '''Gettter of the control vector at time instance i from the optimization
           variable x.'''
        start = self.nz*(self.N - 1) + self.nu*i
        if self.nu == 1:
            return x[start]
        return x[start: start + self.nu]

    def get_state(self, x, i):
        '''Gettter of the state vector at time instance i from the optimization
           variable x.'''
        if i == 0:
            return self.z0
        start = self.nz*(i-1)
        return x[start: start + self.nz]

    def set_state(self, x, z, i):
        '''Setter for a state vector z at time instance i for the optimization
           variable x.'''
        start = self.nz*(i-1)
        x[start: start + self.nz] = z
        return x

    def set_control(self, x, u, i):
        '''Setter for a control vector u at time instance i for the optimization
           variable x.'''
        start = self.nz*(self.N - 1) + self.nu*i
        x[start: start + self.nu] = u
        return x

    def get_beta(self, u):
        '''Getter for the beta value in the control vector u'''
        if self.nu > 1:
            return u[1]
        return u

    def set_beta(self, u, beta):
        '''Setter for the beta value in the control vector u'''
        if self.nu > 1:
            u[1] = beta
        else:
            u = beta
        return u

    def get_model_constraints(self, x):
        '''Checks the optimization variable for violation against model dynamics'''
        f_eq = np.zeros(self.nz*(self.N - 1))
        z_prev = self.z0
        for k in range(self.N - 1):
            curr_u = self.get_control(x, k)
            new_state = self.model_dynamics(z_prev, curr_u)
            f_eq[self.nz*k: self.nz*(k+1)] = new_state - self.get_state(x, k+1)
            z_prev = new_state
        return f_eq

    def get_input_constraints(self, x):
        '''Checks the optimization variable for violations against beta dot max'''
        f_ineq = np.zeros(self.N - 1)
        beta_prev = self.beta0
        for k in range(self.N - 1):
            u = self.get_control(x, k)
            f_ineq[k] = self.beta_dot_max - abs((self.get_beta(u) - beta_prev)/self.Ts)
            beta_prev = self.get_beta(u)
        return f_ineq

    def calc_control(self, z, partial_tracking=False):
        '''Calculate the MPC control given the current state z and the previous
           wheel angle beta. Solves the optimization problem and returns the
           first input step, the evaluated cost, the number of iterations performed
           by the solved and the total solve time.'''
        self.z0 = z
        t_start = time.process_time()
        init_guess = self.calc_init_guess()
        res = opt.minimize(self.cost_function,
                           init_guess,
                           method='SLSQP',
                           constraints=self.constraints,
                           bounds=self.bounds,
                           options={
                               'maxiter': 150,
                               'disp': True,
                               #'ftol': 0.01
                           })
        t_solve = time.process_time() - t_start
        if partial_tracking:
            self.print_partial_progress(res.x, init_guess)
        self._x_opt = res.x
        u_opt = self.get_control(res.x, 0)
        self.beta0 = self.get_beta(u_opt)
        return u_opt, res.fun, res.nit, t_solve

    def calc_init_guess(self):
        '''Generates an initial guess for the solver to start from. Based on two
           solutions:
           1) if there exists no previous solution from an earlier step, call
              generate_new_guess()
           2) if there exists a previous solution, try to base the new initial guess
              from this. Take all points from the previous solution except the first
              one, and then use the model dynamics to generate the new step. If this
              step is feasible, use it. If its not, try to steer away from the
              obstacle and return the first feasible variable.'''

        if self._x_opt is None:
            # No previous solution is available, generate a new one.
            return self.generate_new_guess()

        # create a new optimization vector and copy the elements from the previous one
        x_init = np.zeros((self.nz + self.nu)*(self.N - 1))
        for k in range(self.N - 2):
            _z_opt = self.get_state(self._x_opt, k + 2)
            x_init = self.set_state(x_init, _z_opt, k + 1)
            _u_opt = self.get_control(self._x_opt, k + 1)
            x_init = self.set_control(x_init, _u_opt, k)
        # using the previous input, use the model dynamics to generate a new state
        z_N = self.model_dynamics(_z_opt, _u_opt)
        x_init = self.set_control(x_init, _u_opt, self.N - 2)
        x_init = self.set_state(x_init, z_N, self.N - 1)
        # run through all obstacles in the track
        for obs in self.obstacles:
            if obs.closest_distance(z_N[0], z_N[1]) < self.margin:
                # the last state is inside the margin for the obstacle
                # find which way we should steer to get away from the obstacle
                sgn = np.sign(obs.get_closest_edge_angle(_z_opt[0], _z_opt[1]))
                # iterate backwards through each time step
                for n in range(self.N - 1, 0, -1):
                    z = self.get_state(x_init, n - 1)
                    # set the control for this time step as the maximum allowed
                    # steering angle away from the obstacle
                    u = self.get_control(x_init, n - 1)
                    _u = self.get_control(x_init, n - 2)
                    beta = self.get_beta(_u) + sgn*self.Ts*self.beta_dot_max
                    # TODO: check for beta_max as well.
                    u = self.set_beta(u, beta)
                    # use the new control sequence to generate new states
                    for k in range(self.N - n):
                        z = self.model_dynamics(z, u)
                        x_init = self.set_control(x_init, u, n + k - 1)
                        x_init = self.set_state(x_init, z, n + k)
                        beta = self.get_beta(u) + sgn*self.Ts*self.beta_dot_max
                        u = self.set_beta(u, beta)
                    # check if the last state is outside the obstacle margin
                    if obs.closest_distance(z[0], z[1]) >= self.margin:
                        # if it is, break out of the loop
                        break
                    # else, go one step further back in time and try to steer earlier
                else:
                    # could not steer away in time, no initial feasible solution
                    # could be found
                    raise ValueError('cannot find init guess!')
        return x_init

    def generate_new_guess(self):
        '''Generates an initial guess when there is no previous solution to base
           it from.'''
        x_init = np.zeros((self.nz + self.nu)*(self.N - 1))
        for k in range(self.N - 1):
            new_state = self.model_dynamics(
                self.get_state(x_init, k),
                self.get_control(x_init, k))
            x_init = self.set_state(x_init, new_state, k + 1)
            for obs in self.obstacles:
                if obs.closest_distance(new_state[0], new_state[1]) < self.margin:
                    _z = self.z0
                    _beta = self.beta0
                    _u = self.get_control(x_init, 0)
                    _u = self.set_beta(_u, _beta)
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
                        _u = self.set_beta(_u, _beta)
                        x_init = self.set_control(x_init, _u, n)
                        _z = self.model_dynamics(_z, _u)
                        x_init = self.set_state(x_init, _z, n + 1)
        return x_init

    def print_partial_progress(self, x_opt, x_init):
        '''Plots the initial guess and optimal solution for the current time
           step, including the prediction horizion.'''
        z_opt = np.zeros((self.N, self.nz))
        z_init = np.zeros((self.N, self.nz))
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
            ref_cost = self.xTQx(ref_err, self.Qz)
            curr_u = self.get_control(x, k)
            u_cost = self.xTQx(curr_u, self.Qu)
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
            u_cost = self.xTQx(curr_u, self.Qu)
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
                f_obs[n_obs*(k-1)+i] = obs.closest_distance(curr_z[0], curr_z[1]) - self.margin
        return np.concatenate((f_input, f_obs))

class LinearizedMPC(MPC):

    def __init__(self, Qz, Qu, Qf, Qzf, N):
        super().__init__(Qz, Qu, Qf, N)
        self.nu = 1
        self.H = Qz
        for i in range(self.N-2):
            z1 = np.zeros((self.H.shape[0], self.Qz.shape[1]))
            z2 = np.zeros((self.Qz.shape[0], self.H.shape[1]))
            self.H = np.bmat([
                [self.H, z1],
                [z2, self.Qz]
            ])
        for i in range(self.N-1):
            z1 = np.zeros((self.H.shape[0], self.Qu.shape[1]))
            z2 = np.zeros((self.Qu.shape[0], self.H.shape[1]))
            self.H = np.bmat([
                [self.H, z1],
                [z2, self.Qu]
            ])
        self.h = np.tile(Qzf, self.N-1)
        self.h = np.concatenate((self.h, np.zeros((self.N-1)*self.nu)))

    def cost_function(self, x):
        quadcost = self.xTQx(x, self.H)
        lincost = np.dot(x, self.h)
        return quadcost + lincost

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
