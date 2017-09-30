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

    def closest_distance(self, x, y):
        in_x = self.x_edge_lo <= x and x <= self.x_edge_hi
        in_y = self.y_edge_lo <= y and y <= self.y_edge_hi
        if in_x:
            if in_y:
                # inside
                x_c = self.x_edge_lo + self.x_size/2
                y_c = self.y_edge_lo + self.y_size/2
                dist_x_lo = self.x_edge_lo - x
                dist_x_hi = x - self.x_edge_hi
                dist_y_lo = self.y_edge_lo - y
                dist_y_hi = y - self.y_edge_hi
                return max([dist_x_lo, dist_x_hi, dist_y_lo, dist_y_hi])
            else:
                if y > self.y_edge_hi:
                    return y - self.y_edge_hi
                else:
                    return self.y_edge_lo - y
        else:
            if in_y:
                if x > self.x_edge_hi:
                    return x - self.x_edge_hi
                else:
                    return self.x_edge_lo - x
            else:
                # corners
                if x > self.x_edge_hi:
                    x_vec = x - self.x_edge_hi
                else:
                    x_vec = self.x_edge_lo - x
                if y > self.y_edge_hi:
                    y_vec = y - self.y_edge_hi
                else:
                    y_vec = self.y_edge_lo - y
                return np.sqrt(x_vec**2 + y_vec**2)

    def get_closest_edge_angle(self, x, y):
        y_diff_hi = self.y_edge_hi - y
        y_diff_lo = self.y_edge_lo - y
        x_diff = self.x_edge_lo - x
        angle_hi = np.arctan2(y_diff_hi, x_diff)
        angle_lo = np.arctan2(y_diff_lo, x_diff)
        i = np.argmin([angle_hi, abs(angle_lo)])
        return (angle_hi, angle_lo)[i]

    def get_plot_params(self):
        lower_left = (self.x_edge_lo, self.y_edge_lo)
        return [lower_left, self.x_size, self.y_size]

class Track:

    def __init__(self, o):
        self.x_start = o['x_start']
        self.x_end = o['x_end']
        self.y_edge_hi = o['y_edge_hi']
        self.y_edge_lo = o['y_edge_lo']

    def in_track(self, x, y):
        if self.y_edge_lo < y and y < self.y_edge_hi:
            return 0
        return -1
