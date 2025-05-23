import numpy as np
from shapely.geometry import Point
from road_damage import RoadDamage

class Detection:
    def __init__(self, x: float, y: float, step: int, detected: bool):
        self.x = x
        self.y = y
        self.detected = detected
        self.step = step  # step number when the detection was made

    def __repr__(self):
        return f"<Detection ({self.x:.2f}, {self.y:.2f}) detected={self.detected}>"

    def __str__(self):
        return f"Detection at ({self.x:.2f}, {self.y:.2f}) with detection={self.detected}"

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
        """
        Simulate the detection result based on the actual occupancy.

        :param actual_occupied:
        :return:
        """
        if actual_occupied:
            # possiblity of true positive
            return np.random.rand() < self.p_true
        else:
            # possibility of false positive
            return np.random.rand() < self.p_false

    def detect_damage_position(self, step: int, x: float, y: float):
        """
        Returns the Damage instance if the (x,y) observation falls within any damage shape,
        otherwise None.
        """

        x_est, y_est = self.sample_gps(x, y)

        if self.damage_model is None:
            # no damage model provided
            return Detection(x_est, y_est, step, self.detect_result(False) )
        # check each damage shape
        for damage in self.damage_model.all_damages():
            if damage.contains(x, y):
                # sample GPS noise
                return Detection(x_est, y_est, step, self.detect_result(True) )

        # no damage detected but the sensor still returns a position
        return Detection(x_est, y_est, step, self.detect_result(False))