import numpy as np

class Obstacle:

    def __init__(self, o):
        self.x_size = o['x_size']
        self.y_size = o['y_size']
        self.x_edge_lo = o['x'] - o['x_size']/2
        self.x_edge_hi = o['x'] + o['x_size']/2
        self.y_edge_lo = o['y'] - o['y_size']/2
        self.y_edge_hi = o['y'] + o['y_size']/2

    def collision(self, x, y):
        in_x = self.x_edge_lo <= x and x <= self.x_edge_hi
        in_y = self.y_edge_lo <= y and y <= self.y_edge_hi
        if in_x and in_y:
            return -1
        return 0

    def get_closest_edge_angle(self, x, y):
        y_diff_hi = self.y_edge_hi - y
        y_diff_lo = self.y_edge_lo - y
        x_diff = self.x_edge_lo - x
        angle_hi = np.arctan2(y_diff_hi, x_diff)
        angle_lo = np.arctan2(y_diff_lo, x_diff)
        i = np.argmin([angle_hi, abs(angle_lo)])
        return [angle_hi, angle_lo][i]

    def get_plot_params(self):
        lower_left = (self.x_edge_lo, self.y_edge_lo)
        return [lower_left, self.x_size, self.y_size]
