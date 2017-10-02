import numpy as np
import time

class SimMaster:

    def __init__(self, Ts):
        self.Ts = Ts

    def run_sim(self, track, car_model, controller):
        t_vec = np.array([None])
        z_vec = np.array([None, None, None, None])
        u_vec = np.array([None, None])
        j_vec = np.array([None])
        i_vec = np.array([None, None])
        i = 0
        while True:
            t_start = time.process_time()
            u, j, itr = controller.calc_control(car_model.z)
            t_solve = time.process_time() - t_start
            t_vec = np.vstack((t_vec, i*self.Ts))
            z_vec = np.vstack((z_vec, car_model.z))
            u_vec = np.vstack((u_vec, u))
            j_vec = np.vstack((j_vec, j))
            i_vec = np.vstack((i_vec, [t_solve, itr]))
            print('Time {:1.1f} solved'.format(i*self.Ts))
            print('Curr state: {0}'.format(car_model.z))
            if car_model.z[0] >= controller.x_goal:
                break
            car_model.update_state(u)
            i = i + 1

        return t_vec[1:], z_vec[1:, :], u_vec[1:, :], j_vec[1:], i_vec[1:, :]
