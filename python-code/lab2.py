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
N = 15
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
t, z, u, j = car.run_sim(mpc, ref)

car_pos = [(x, y) for x, y in zip(z[:, 0], z[:, 1])]
x = z[:, 0]
y = z[:, 1]
fig = plt.figure()
ax_states = fig.add_subplot(2, 2, 1)
ax_states.plot(t, x, 'b')
ax_states.plot(t, y, 'r')
ax_xy = fig.add_subplot(2, 2, 2)
ax_xy.plot(x, y)
[ax_xy.add_patch(
    ptch.Rectangle(
        *obs.get_plot_params(),
        facecolor='red',
        hatch='/'
    )
) for obs in obstacles]
ax_xy.set_xlim([0, 50])
ax_xy.set_ylim([-8, 8])
#plt.plot(ref_x, ref_y)
ax_cost = fig.add_subplot(2, 2, 3)
ax_cost.plot(t, j)
ax_input = fig.add_subplot(2, 2, 4)
ax_input.plot(t, u[:, 0])
ax_input.plot(t, u[:, 1])
plt.show()
