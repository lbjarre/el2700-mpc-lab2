from vehiclemodel import CarModel
from gui import GUI
from controllers import MPC
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('python-code/carmodels.yaml') as input_file:
    car_models = yaml.safe_load(input_file)

car = CarModel(car_models['car1'])

Q1 = np.identity(2)
Q2 = 0.001*np.identity(2)
Qf = Q1
N = 5
ref_y = np.sin(2*np.pi*0.1*np.arange(0, 10, 0.1))
ref_x = np.linspace(0, 10, len(ref_y))
ref = np.transpose(np.array([ref_y, ref_x]))

nmpc = MPC(Q1, Q2, Qf, N, ref, car)
t, z, u, j = car.run_sim(nmpc)

car_pos = [(x, y) for x, y in zip(z[:, 0], z[:, 1])]
x = z[:, 0]
y = z[:, 1]
plt.plot(t, x, 'b')
plt.plot(t, y, 'r')
plt.plot(t, ref_x, 'b--')
plt.plot(t, ref_y, 'r--')
plt.show()
plt.plot(t, j)
plt.show()
plt.plot(t, u[:, 0])
plt.plot(t, u[:, 1])
plt.show()
