from trackmodel import Obstacle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

obs = Obstacle({
    'x': 0,
    'y': 0,
    'x_size': 2,
    'y_size': 6
})

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
z = np.vectorize(obs.closest_distance)(x, y)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(x, y, z)
plt.show()

with open('data/closest_distance.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['x', 'y', 'z'])
    iterlist = zip(np.ndarray.flatten(x), np.ndarray.flatten(y), np.ndarray.flatten(z))
    for r in iterlist:
        writer.writerow(r)
