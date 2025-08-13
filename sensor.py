import numpy as np
from shapely.geometry import Point
from road_damage import RoadDamage

class Detection:
    def __init__(self, x: float, y: float, step: int, detected: bool, type: str, eval: str = "na"):
        self.x = x
        self.y = y
        self.detected = detected
        self.step = step  # step number when the detection was made
        self.type = type
        self.eval = eval # Evaluation result, tp, fn, fp, tn, or na

    def __repr__(self):
        return f"<Detection ({self.x:.2f}, {self.y:.2f}) detected={self.detected}>"

    def __str__(self):
        return f"Detection at ({self.x:.2f}, {self.y:.2f}) with detection={self.detected}"

class VehicleSensor:
    def __init__(self, prob_dict, gps_sigma=5.0,
                 max_range=None, damage_model: RoadDamage = None):
        self.prob_dict = prob_dict
        self.anomaly_types = [
            k for k in prob_dict.keys() if k != "mild_road"]
        self.gps_sigma = gps_sigma
        self.max_range = max_range
        self.damage_model = damage_model

    def sample_gps(self, x, y):
        x_noisy = x + np.random.normal(0, self.gps_sigma)
        y_noisy = y + np.random.normal(0, self.gps_sigma)
        return x_noisy, y_noisy

    def detect_result(self, actual_occupied, road_anomaly_type: str):
        """
        Simulate the detection result based on the actual occupancy.

        :param actual_occupied:
        :return:
        """
        # when actual occupancy is True, it means the road anomaly is present and must be in the key of prob_dict

        if actual_occupied:
            # possibility of true positive
            p_true = self.prob_dict[road_anomaly_type]['tp']
            if np.random.rand() < p_true:
                return True, road_anomaly_type, 'tp'
            else:
                # if the true positive fails, it is a false negative
                return False, "na", 'fn'
        else:
            # randomly select one road anomaly type from the probability dictionary
            # possibility of false positive (false alarm) of driving on a mild road
            p_false = self.prob_dict['mild_road']['fp']
            if np.random.rand() < p_false:
                road_anomaly_type = np.random.choice(self.anomaly_types) # assume each of the K road anomaly types has the same chance of being detected
                return True, road_anomaly_type, 'fp'
            else:
                # if the false positive fails, it is a true negative
                return False, "na", 'tn'


    def detect_damage_position(self, step: int, x: float, y: float):
        """
        Returns the Damage instance if the (x,y) observation falls within any damage shape,
        otherwise None.
        """

        x_est, y_est = self.sample_gps(x, y)

        if self.damage_model is None:
            # no damage model provided
            detected = self.detect_result(False, "na")
            if detected[0]: # if the damage is detected, it is a false positive
                return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])
            else: # if the damage is not detected, it is a true negative
                return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])

        # check each damage shape
        for damage in self.damage_model.all_damages():
            if damage.contains(x, y):
                # sample GPS noise
                detected = self.detect_result(True, damage.type)
                if detected[0]:  # if the damage is detected
                    return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])
                else:  # if the damage is not detected, it is a false negative
                    return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])

        # no damage detected but the sensor still returns a position
        detected = self.detect_result(False, "na")
        if detected[0]:  # if the damage is detected, it is a false positive
            return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])
        else:  # if the damage is not detected, it is a true negative
            return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])

    def detect_damage_travel_position(self, step: int, last_x: float, last_y: float, x: float, y: float) -> Detection:
        """
        Returns the Damage instance if the travel path from (last_x, last_y) to (x,y)
        intersects with any damage shape, otherwise None.
        """
        x_est, y_est = self.sample_gps(x, y)  # TODO new feature... generate the car with lateral movement distribution?

        # check each damage shape if it detects the travel path
        for damage in self.damage_model.all_damages():
            # the travel path intersects with the damage width, so it might hit the damage
            if damage.intersects_travel_path(last_x, last_y, x, y):
                hit_probability = damage.probability
                if np.random.rand() < hit_probability:
                    # if the damage is hit, return detection
                    detected = self.detect_result(True, damage.type)
                    return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])



        # if the damage is not hit, check the possibility of a false positive

        detected = self.detect_result(False, "na")
        return Detection(x_est, y_est, step, detected[0], detected[1], detected[2])



