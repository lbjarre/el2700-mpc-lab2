from vehiclemodel import CarModel
from trackmodel import Obstacle, Track
from controllers import RefTrackMPC, ObstacleAvoidMPC
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

with open('python-code/models.yaml') as input_file:
    models = yaml.safe_load(input_file)

car = CarModel(models['car'])
obstacles = [Obstacle(o) for o in models['obstacles']]
track = Track(models['track'])

Q1 = 10*np.identity(2)
Q2 = 0.01*np.identity(2)
Qf = 1
N = 9
Ts = 0.1
x_goal = 50
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, Ts)
ref_y = 2*np.sin(2*np.pi*0.1*t)
ref_x = np.linspace(0, 100, len(t))
ref = np.transpose(np.array([ref_x, ref_y]))

# mpc = RefTrackMPC(Q1, Q2, Qf, N, car)
mpc = ObstacleAvoidMPC(Q1, Q2, Qf, N, car, obstacles, track)
t, z, u, j, i = car.run_sim(mpc, ref)

fig = plt.figure()

ax_xy = fig.add_subplot(3, 2, 1)
[ax_xy.add_patch(
    ptch.Rectangle(
        *obs.get_plot_params(),
        facecolor='red',
        hatch='/'
    )
) for obs in obstacles]
ax_xy.plot(z[:, 0], z[:, 1])
ax_xy.set_title('XY-plot')
ax_xy.set_xlabel('x [m]')
ax_xy.set_ylabel('y [m]')
ax_xy.set_xlim([0, 50])
ax_xy.set_ylim([-8, 8])

ax_states = fig.add_subplot(3, 2, 3)
[ax_states.plot(t, z[:, k]) for k in range(4)]
ax_states.set_title('States')
ax_states.set_xlabel('Time [s]')
ax_states.set_ylabel('z_n')

ax_input = fig.add_subplot(3, 2, 4)
ax_input.plot(t, u[:, 0])
ax_input.plot(t, u[:, 1])
ax_input.set_title('Inputs')
ax_input.set_xlabel('Time [s]')
ax_input.set_ylabel('u_n')

ax_cost = fig.add_subplot(3, 2, 5)
ax_cost.plot(t, j)
ax_cost.set_title('Cost function')
ax_cost.set_xlabel('Time [s]')
ax_cost.set_ylabel('J_n')

ax_time = fig.add_subplot(3, 2, 6)
ax_time.plot(t, i[:, 0])
ax_time.plot(t, i[:, 1])
ax_time.set_title('Computation')
ax_time.set_xlabel('Time [s]')
ax_time.set_ylabel('Solve time, iterations')

plt.show()
