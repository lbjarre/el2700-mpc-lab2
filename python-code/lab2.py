from vehiclemodel import CarModel
from gui import GUI
from controllers import MPC
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('python-code/carmodels.yaml') as input_file:
    car_models = yaml.safe_load(input_file)

car = CarModel(car_models['car1'])

Q1 = 10*np.identity(2)
Q2 = 0.01*np.identity(2)
Qf = Q1
N = 5
Ts = 0.1
t_start = 0
t_end = 10
t = np.arange(t_start, t_end, Ts)
ref_y = np.linspace(8, 13, len(t))
ref_x = np.linspace(0, 100, len(t))
ref = np.transpose(np.array([ref_x, ref_y]))

nmpc = MPC(Q1, Q2, Qf, N, car)
t, z, u, j = car.run_sim(nmpc, ref)

car_pos = [(x, y) for x, y in zip(z[:, 0], z[:, 1])]
x = z[:, 0]
y = z[:, 1]
plt.plot(t, x, 'b')
plt.plot(t, y, 'r')
plt.plot(t, ref_x, 'b--')
plt.plot(t, ref_y, 'r--')
plt.show()
plt.plot(x, y)
plt.plot(ref_x, ref_y)
plt.show()
plt.plot(t, j)
plt.show()
plt.plot(t, u[:, 0])
plt.plot(t, u[:, 1])
plt.show()
