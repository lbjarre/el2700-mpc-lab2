import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

class MPC:

    def __init__(self, Q1, Q2, Qf, N, car_model, obstacles=[], track=None):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Qf = Qf
        self.N = N
        self.update_functions = car_model.update_functions
        self.a_max = car_model.a_max
        self.beta_max = car_model.beta_max
        self.beta_dot_max = car_model.beta_dot_max
        self.Ts = car_model.Ts
        self.obstacles = obstacles
        self.x_goal = track.x_end
        self.track = track
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
        z_bound = (
            (-np.inf, np.inf),
            (track.y_edge_lo, track.y_edge_hi),
            (-np.inf, np.inf),
            (-np.inf, np.inf)
        )
        u_bound = (
            (-self.a_max, self.a_max),
            (-self.beta_max, self.beta_max)
        )
        self.bounds = np.vstack((np.tile(z_bound, (self.N - 1, 1)), np.tile(u_bound, (self.N - 1, 1))))

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
        return x[start:start + 2]

    def get_state(self, x, i):
        '''Gets u(i) from the optimization variable x'''
        if i == 0:
            return self.z0
        start = 4*(i-1)
        return x[start:start + 4]

    def get_model_constraints(self, x):
        f_eq = np.zeros(4*(self.N - 1))
        z_prev = self.z0
        for k in range(self.N - 1):
            curr_u = self.get_control(x, k)
            new_state = self.update_functions(z_prev, curr_u)
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

    def calc_control(self, z, beta, reference):
        '''Calculate the MPC control given the current state z and the previous
           wheel angle beta. Solves the optimization problem and returns the
           first input step and the evaluated cost.'''
        self.z0 = z
        self.beta0 = beta
        self.reference = reference
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
        # if not res.success: print('No viable solution found!')
        return self.get_control(res.x, 0), res.fun

    def calc_init_guess(self):
        '''Calculates an initial guess for the optimization problem based on
           the input state and the wheel angle. Lets the system dynamics predict
           the next states given no acceleration and the same wheel angle, which
           is guaranteed to be a feasible solution.'''
        a = 0.0
        u = np.tile([a, self.beta0], (self.N - 1, 1))
        while True:
            z = np.zeros((self.N - 1, 4), dtype=np.float64)
            z_prev = self.z0
            for k in range(self.N - 1):
                z[k, :] = self.update_functions(z_prev, u[k, :])
                # check if the new state causes any collisions
                for obs in self.obstacles:
                    if obs.collision(z[k, 0], z[k, 1]):
                        angle = obs.get_closest_edge_angle(self.z0[0], self.z0[1])
                        beta_desired = angle - self.z0[3]
                        beta_dot_desired = (beta_desired - self.beta0)/self.Ts
                        sgn = np.sign(beta_dot_desired)
                        beta_prev = self.beta0
                        for n in range(self.N - 1):
                            sel = np.argmin([abs(beta_dot_desired), self.beta_dot_max])
                            beta_prev = [beta_desired, beta_prev + sgn*self.Ts*self.beta_dot_max][sel]
                            u[n, :] = np.array([a, beta_prev])
                            beta_desired = beta_desired - beta_prev
                            if sel == 0:
                                break
                        else:
                            a = -1
                        u[n + 1:, :] = np.tile([0, beta_prev], (self.N - n - 2, 1))
                        break
                else:
                    # no collisions detected, add state to initial guess
                    z_prev = z[k, :]
                    continue
                # collision detected, reset inital guess and retry with new u
                break
            else:
                # no collisions detected for all n, exit the outer loop
                break
        # flatten z and u and stack into the optimization variable
        z_flat = np.ndarray.flatten(z)
        u_flat = np.ndarray.flatten(u)
        return np.concatenate((z_flat, u_flat))

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
