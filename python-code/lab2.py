from vehiclemodel import CarModel
from trackmodel import Obstacle, Track
from controllers import RefTrackMPC, ObstacleAvoidMPC
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import csv

with open('python-code/models.yaml') as input_file:
    models = yaml.safe_load(input_file)

car = CarModel(models['car'])
obstacles = [Obstacle(o) for o in models['obstacles']]
track = Track(models['track'])

Q1 = 10*np.identity(2)
Q2 = 0.01*np.identity(2)
Qf = 1
N = 10
ref = np.zeros((100, 4))

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
ax_time.plot(t, 1000*i[:, 0])
ax_time.plot(t, i[:, 1])
ax_time.set_title('Computation')
ax_time.set_xlabel('Time [s]')
ax_time.set_ylabel('Solve time [ms], iterations [#]')

plt.show()

print(t.shape)

with open('data/nmpc_N10.csv', 'w+') as outfile:
    writer = csv.writer(outfile, delimiter=' ')
    writer.writerow(['t', 'x', 'y', 'v', 'psi', 'a', 'beta', 'j', 'tsolve', 'niter'])
    iterlist = zip(
        t[:, 0],
        z[:, 0],
        z[:, 1],
        z[:, 2],
        z[:, 3],
        u[:, 0],
        u[:, 1],
        j[:, 0],
        i[:, 0],
        i[:, 1]
    )
    for r in iterlist:
        writer.writerow(r)
