import numpy as np
import scipy.optimize as opt

class MPC:

    def __init__(self, Q1, Q2, Qf, N, update_functions, reference):
        self.Q1 = Q1
        self.Q2 = Q2
        self.Qf = Qf
        self.N = N
        self.update_functions = update_functions
        self.reference = reference

        self.constraints = (
            {
                'type': 'eq',
                'fun': self.equality_constraints
            }
        )

    def cost_function(self, x):
        J = 0;
        for k in range(self.N-1):
            ref_err = self.reference[k, :] - self.get_state(x, k)[0:2]
            ref_cost = self.xTQx(ref_err, self.Q1)
            curr_u = self.get_control(x, k)
            u_cost = self.xTQx(curr_u, self.Q2)
            J = J + ref_cost + u_cost
        ref_err = self.reference[self.N, :] - self.get_state(x, self.N)[0:2]
        term_cost = self.xTQx(ref_err, self.Qf)
        return J + term_cost

    def xTQx(self, x, Q):
        return np.dot(np.dot(np.transpose(x), Q), x)

    def equality_constraints(self, x):
        f_eq = np.zeros(4*self.N)
        z_prev = self.z0
        for i in range(self.N):
            curr_u = self.get_control(x, i)
            new_state = self.update_functions(z_prev, curr_u)
            f_eq[4*i: 4*(i+1)] = new_state - self.get_state(x, i+1)
            z_prev = new_state
        return f_eq

    def get_control(self, x, i):
        start = 4*(self.N + 1) + 2*i
        return x[start:start + 2]

    def get_state(self, x, i):
        start = 4*i
        return x[start:start + 4]

    def calc_control(self, z):
        self.z0 = z
        init_guess = np.zeros(4*(self.N+1) + 2*self.N)
        res = opt.minimize(self.cost_function,
                           init_guess,
                           method='SLSQP',
                           constraints=self.constraints)
        return self.get_control(res.x, 0)
