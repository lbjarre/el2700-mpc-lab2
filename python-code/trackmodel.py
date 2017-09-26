

class Obstacle:

    def __init__(self, o):
        self.x_edge_lo = o['x'] - o['x_size']/2
        self.x_edge_hi = o['x'] + o['x_size']/2
        self.y_edge_lo = o['y'] - o['y_size']/2
        self.y_edge_hi = o['y'] + o['y_size']/2

    def collision(self, x, y):
        in_x = self.x_edge_lo <= x and x <= self.x_edge_hi
        in_y = self.y_edge_lo <= y and y <= self.y_edge_hi
        return -1 if (in_x and in_y) else 0