import numpy as np
from shapely.geometry import Point
from road_damage import RoadDamage

class VehicleSensor:
    def __init__(self, p_true=0.9, p_false=0.05, gps_sigma=5.0,
                 max_range=None, damage_model: RoadDamage = None):
        self.p_true = p_true
        self.p_false = p_false
        self.gps_sigma = gps_sigma
        self.max_range = max_range
        self.damage_model = damage_model

    def sample_gps(self, x, y):
        x_noisy = x + np.random.normal(0, self.gps_sigma)
        y_noisy = y + np.random.normal(0, self.gps_sigma)
        return x_noisy, y_noisy

    def detect_result(self, actual_occupied):
        if actual_occupied:
            return np.random.rand() < self.p_true
        else:
            return np.random.rand() < self.p_false

    def detect_damage_position(self, x: float, y: float):
        """
        Returns the Damage instance if the (x,y) observation falls within any damage shape,
        otherwise None.
        """
        if self.damage_model is None:
            return None, None, False
        # check each damage shape
        for damage in self.damage_model.all_damages():
            if damage.contains(x, y):
                # sample GPS noise
                x_est, y_est = self.sample_gps(x, y)
                return x_est, y_est, True

        x_est, y_est = self.sample_gps(x, y)
        # no damage detected but the sensor still returns a position
        return x_est, y_est, False