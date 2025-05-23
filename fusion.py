from grid import OccupancyGrid
from sensor import VehicleSensor
import numpy as np

class FusionEngine:
    def __init__(self, grid: OccupancyGrid, sensor_model: VehicleSensor):
        self.grid = grid
        self.sensor = sensor_model

    def update(self, detections):
        """
        detections: list of tuples (x, y, detected)
        where (x,y) is noisy sensor location, detected is bool.
        """
        # convert GPS coordinates to grid cell indices


        for detection in detections:
            x_gps, y_gps = detection.x, detection.y
            detected = detection.detected
            cell = self.grid.world_to_grid(x_gps, y_gps)
            if cell is None:
                continue
            i, j = cell
            # select measurement probability p_z
            self.grid.update_cell(i, j, self.sensor.p_true, self.sensor.p_false, detected)

    def batch_update(self, log_path: str):
        """
        Batch update the grid by reading a log file in txt
        Example of each line is saved as:

        6 2018.4998970206116 973.9165659938301 True

        """
        self.grid.batch_update(log_path, self.sensor)