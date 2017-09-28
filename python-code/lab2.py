from vehiclemodel import CarModel
from trackmodel import Obstacle
from controllers import RefTrackMPC, ObstacleAvoidMPC
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

with open('python-code/models.yaml') as input_file:
    models = yaml.safe_load(input_file)

car = CarModel(models['car'])
obstacles = [Obstacle(o) for o in models['obstacles']]
#obstacles = []

Q1 = 10*np.identity(2)
Q2 = 0.01*np.identity(2)
Qf = 1
N = 5
Ts = 0.1
x_goal = 50
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, Ts)
ref_y = 2*np.sin(2*np.pi*0.1*t) + 8
ref_x = np.linspace(0, 100, len(t))
ref = np.transpose(np.array([ref_x, ref_y]))

# mpc = RefTrackMPC(Q1, Q2, Qf, N, car)
mpc = ObstacleAvoidMPC(Q1, Q2, Qf, N, car, obstacles, x_goal)
t, z, u, j = car.run_sim(mpc, ref)

car_pos = [(x, y) for x, y in zip(z[:, 0], z[:, 1])]
x = z[:, 0]
y = z[:, 1]
plt.plot(t, x, 'b')
plt.plot(t, y, 'r')
#plt.plot(t, ref_x, 'b--')
#plt.plot(t, ref_y, 'r--')
plt.show()
plt.plot(x, y)
ax = plt.gca()
[
    ax.add_patch(
        ptch.Rectangle(
            *obs.get_plot_params(),
            facecolor='red',
            hatch='/'
        )
    ) for obs in obstacles]
#plt.plot(ref_x, ref_y)
plt.show()
plt.plot(t, j)
plt.show()
plt.plot(t, u[:, 0])
plt.plot(t, u[:, 1])
plt.show()
