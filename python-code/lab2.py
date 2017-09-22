from vehiclemodel import CarModel
import numpy as np
import yaml
import matplotlib.pyplot as plt

with open('python-code/carmodels.yaml') as input_file:
    car_models = yaml.safe_load(input_file)

car = CarModel(car_models['car1'])

time = np.arange(0, 10, 0.1)
inputs = np.random.normal(0, 10, (len(time), 2))
outputs = np.zeros((len(time), 4))

for i, t in enumerate(time):
    car.update_state(inputs[i, :])
    outputs[i, :] = car.z

plt.plot(time, outputs)
plt.show()
