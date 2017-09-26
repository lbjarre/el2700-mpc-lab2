import numpy as np
import scipy.optimize as opt

class MPC:

    def __init__(self, Q1, Q2, Qf, N, car_model):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Qf = Qf
        self.N = N
        self.update_functions = car_model.update_functions
        self.a_max = car_model.a_max
        self.beta_max = car_model.beta_max
        self.beta_dot_max = car_model.beta_dot_max
        self.Ts = car_model.Ts
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
        start = 4*(self.N + 1) + 2*i
        return x[start:start + 2]

    def get_state(self, x, i):
        '''Gets u(i) from the optimization variable x'''
        start = 4*i
        return x[start:start + 4]

    def get_model_constraints(self, x):
        f_eq = np.zeros(4*self.N)
        z_prev = self.z0
        for k in range(self.N):
            curr_u = self.get_control(x, k)
            new_state = self.update_functions(z_prev, curr_u)
            f_eq[4*k: 4*(k+1)] = new_state - self.get_state(x, k+1)
            z_prev = new_state
        return f_eq

    def get_input_constraints(self, x):
        f_ineq = np.zeros(3*self.N)
        beta_prev = self.beta0
        for k in range(self.N):
            curr_u = self.get_control(x, k)
            f_ineq[3*k] = self.a_max - abs(curr_u[0])
            f_ineq[3*k+1] = self.beta_max - abs(curr_u[1])
            f_ineq[3*k+2] = self.beta_dot_max - abs((curr_u[1] - beta_prev)/self.Ts)
            beta_prev = curr_u[1]
        return f_ineq

    def calc_control(self, z, beta, reference):
        '''Calculate the MPC control given the current state z and the previous
           wheel angle beta. Solves the optimization problem and returns the
           first input step and the evaluated cost.'''
        self.z0 = z
        self.beta0 = beta
        self.reference = reference
        init_guess = self.calc_init_guess(z, beta)
        res = opt.minimize(self.cost_function,
                           init_guess,
                           method='SLSQP',
                           constraints=self.constraints)
        if not res.success: print('No viable solution found!')
        return self.get_control(res.x, 0), res.fun

    def calc_init_guess(self, z, beta):
        '''Calculates an initial guess for the optimization problem based on
           the input state and the wheel angle. Lets the system dynamics predict
           the next states given no acceleration and the same wheel angle, which
           is guaranteed to be a feasible solution.'''
        z_init_guess = z
        u = np.array([0, beta])
        for k in range(self.N):
            z = self.update_functions(z, u)
            z_init_guess = np.concatenate((z_init_guess, z))
        return np.concatenate((z_init_guess, np.tile(u, self.N)))

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

    def __init__(self, Q1, Q2, Qf, N, car_model, obstacles, x_goal):
        super().__init__(Q1, Q2, Qf, N, car_model)
        self.obstacles = obstacles
        self.x_goal = x_goal

    def cost_function(self, x):
        J = 0
        for k in range(self.N-1):
            curr_u = self.get_control(x, k)
            J = J + self.xTQx(curr_u, self.Q2)
        term_z = self.get_state(x, self.N)
        dist_to_goal = self.x_goal - term_z[0]
        if dist_to_goal > 0:
            return J + dist_to_goal*self.Qf
        return J

    def equality_constraints(self, x):
        return self.get_model_constraints(x)

    def inequality_constraints(self, x):
        f_input = self.get_input_constraints(x)
        n_obs = len(self.obstacles)
        f_obs = np.zeros(n_obs*self.N)
        for k in range(self.N):
            curr_z = self.get_state(x, k)
            for i, obs in enumerate(self.obstacles):
                f_obs[n_obs*k: n_obs*k+i] = obs.collision(curr_z[0], curr_z[1])
        return np.concatenate((f_input, f_obs))
