#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sumolib

from grid import OccupancyGrid
from sensor import VehicleSensor
from sumo_interface import SumoInterface
from fusion import FusionEngine
from road_damage import RoadDamage

# --- USER CONFIG ---
NET_FILE    = 'scenario/A2_GO2GW_v2_MM.net.xml'
ROUTE_FILE  = 'scenario/A2Graz.rou.xml'
SUMO_CMD    = ['sumo', '-n', NET_FILE, '-r', ROUTE_FILE, '--step-length', '1.0']

# If you want “damage” on specific edges, list them here:
DAMAGE_EDGE_IDS = ['-4001.0.00', '-4002.0.00', '-5004.0.00']

# Grid specs
X_MIN, X_MAX = -1000, 1000
Y_MIN, Y_MAX = -1000, 1000
RESOLUTION   = 1.0
PRIOR        = 0.1

# Sensor specs
P_TRUE       = 0.90
P_FALSE      = 0.05
GPS_SIGMA    = 5.0

# Simulation length
SIM_STEPS    = 3600
# -------------------


def run():
    # 1) Load road‐damage shapes
    damage_model = RoadDamage(
        net_file=NET_FILE,
        edge_ids=DAMAGE_EDGE_IDS,
        radius=1.0  # buffer radius around each damage point [m]
    )
    print(f"Loaded damage model: {damage_model}")
    for dmg in damage_model.all_damages():
        print(" ", dmg)

    # 2) Init occupancy grid, sensor & fusion engine
    grid = OccupancyGrid(X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, PRIOR)
    sensor_model = VehicleSensor(
        p_true=0.9,
        p_false=0.05,
        gps_sigma=5.0,
        max_range=None,
        damage_model=damage_model
    )
    fusion_eng = FusionEngine(grid, sensor_model)

    # 3) Launch SUMO
    sumo = SumoInterface(SUMO_CMD)

    # 4) Simulation loop
    for step in range(SIM_STEPS):
        sumo.step()
        detections = []

        for vid, (x_true, y_true) in sumo.get_vehicle_positions().items():
            # only passenger vehicles
            if sumo.get_vehicle_type(vid) != 'PasVeh':
                continue

            hit = sensor_model.detect_damage_position(x_true, y_true)
            if hit:
                # hit is an instance of Damage
                loc = hit.location
                print(f"[step {step}] Hit damage on edge {hit.edge_id} at ({loc.x:.1f}, {loc.y:.1f})")
                detections.append(hit)

        fusion_eng.update(detections)

    # 5) Tear down SUMO
    sumo.close()

    # 6) Show final occupancy map
    probmap = grid.get_probability_map()
    plt.figure(figsize=(8,8))
    plt.imshow(
        probmap,
        origin='lower',
        extent=(X_MIN, X_MAX, Y_MIN, Y_MAX)
    )
    plt.colorbar(label='P(occupied)')
    plt.title('Bayesian Occupancy Grid')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.show()

    # idea: visualize the measurements on the map
    # idea: add different severity class to the damage model


if __name__ == "__main__":
    run()
