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
        for x_gps, y_gps, detected in detections:
            cell = self.grid.world_to_grid(x_gps, y_gps)
            if cell is None:
                continue
            i, j = cell
            # select measurement probability p_z
            p = self.sensor.p_true if detected else self.sensor.p_false
            self.grid.update_cell(i, j, p)