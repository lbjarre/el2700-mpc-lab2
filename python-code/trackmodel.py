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
        '''Returns True if the given (x, y) is in the obstacle, else False.'''
        in_x = self.x_edge_lo <= x and x <= self.x_edge_hi
        in_y = self.y_edge_lo <= y and y <= self.y_edge_hi
        if in_x and in_y:
            return True
        return False

    def closest_distance(self, x, y):
        '''Returns the closest distance to an edge of the obstacle from the given
           (x, y). Distances from points (x, y) inside the obstacles are negative.'''
        in_x = self.x_edge_lo <= x and x <= self.x_edge_hi
        in_y = self.y_edge_lo <= y and y <= self.y_edge_hi
        x_dists = [self.x_edge_lo - x, x - self.x_edge_hi]
        y_dists = [self.y_edge_lo - y, y - self.y_edge_hi]
        x_sel = x_dists[np.argmin(np.abs(x_dists))]
        y_sel = y_dists[np.argmin(np.abs(y_dists))]
        if not (in_x or in_y):
            return np.sqrt(x_sel**2 + y_sel**2)
        if in_x:
            if in_y:
                return max([x_sel, y_sel])
            return y_sel
        return x_sel

    def get_closest_edge_angle(self, x, y):
        '''Returns the angle to the closest corner of the obstacle from the given
           point (x, y).'''
        y_diff_hi = self.y_edge_hi - y
        y_diff_lo = self.y_edge_lo - y
        x_diff = self.x_edge_lo - x
        angle_hi = np.arctan2(y_diff_hi, x_diff)
        angle_lo = np.arctan2(y_diff_lo, x_diff)
        i = np.argmin([angle_hi, abs(angle_lo)])
        return (angle_hi, angle_lo)[i]

    def get_plot_params(self):
        '''Returns params for the Patches library to plot the obstacle.'''
        lower_left = (self.x_edge_lo, self.y_edge_lo)
        return [lower_left, self.x_size, self.y_size]

class Track:

    def __init__(self, params):
        self.x_start = params['x_start']
        self.x_end = params['x_end']
        self.y_edge_hi = params['y_edge_hi']
        self.y_edge_lo = params['y_edge_lo']
        self.obstacles = [Obstacle(o) for o in params['obstacles']]

    def in_track(self, x, y):
        if self.y_edge_lo < y and y < self.y_edge_hi:
            return 0
        return -1
