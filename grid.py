import numpy as np

class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=1.0, prior=0.1):
        """
        Initialize an occupancy grid covering [x_min, x_max] x [y_min, y_max]
        at the given resolution (meters per cell) with a flat prior.
        """
        self.resolution = resolution
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.width = int(np.ceil((x_max - x_min) / resolution))
        self.height = int(np.ceil((y_max - y_min) / resolution))
        # log-odds representation
        p = prior
        self.log_odds = np.full((self.height, self.width), self._prob_to_logodds(p))

    def _prob_to_logodds(self, p):
        return np.log(p / (1.0 - p))

    def _logodds_to_prob(self, l):
        return 1.0 - 1.0 / (1.0 + np.exp(l))

    def world_to_grid(self, x, y):
        """
        Convert world (x,y) to grid indices (i,j).
        """
        i = int((y - self.y_min) / self.resolution)
        j = int((x - self.x_min) / self.resolution)
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return None
        return i, j

    def update_cell(self, i, j, p_z):
        """
        Update cell (i,j) with measurement probability p_z using Bayesian log-odds.
        p_z: P( z | occupied ) / P( z | free )
        """
        l_prior = self.log_odds[i, j]
        l_z = np.log(p_z / (1.0 - p_z))
        self.log_odds[i, j] = l_prior + l_z

    def get_probability_map(self):
        """
        Return the grid as probabilities.
        """
        return self._logodds_to_prob(self.log_odds)