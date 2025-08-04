#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sumolib
import numpy as np
import json
from grid import OccupancyGrid
from sensor import VehicleSensor, Detection
from sumo_interface import SumoInterface
from fusion import FusionEngine
from road_damage import RoadDamage
from typing import Dict, List
from utilities import load_road_anomaly_metrics, gen_damage_area
import traci

# --- USER CONFIG ---
# NET_FILE = 'scenario/Graz_A2/A2_GO2GW_v2_MM.net.xml'
# ROUTE_FILE = 'scenario/Graz_A2/A2Graz.rou.xml'
# ADDITIONAL_FILE = 'scenario/Graz_A2/road_damage.add.xml'
# PROBABILITY_FILE = 'scenario/Graz_A2/road_anomaly_probabilities.json'

SCENARIO_NAME = 'brussel_rural'
NET_FILE = 'scenario/brussel_rural/osm_withProjParam.net.xml'
SUMOCFG_FILE = 'scenario/brussel_rural/osm.sumocfg'
ROUTE_FILE = 'scenario/brussel_rural/osm.rou.xml'
ADDITIONAL_FILE = 'scenario/brussel_rural/potholes.add.xml'
PROBABILITY_FILE = 'data/brussel_rural/road_anomaly_probabilities.json'
ROAD_ANOMALY_DETECTION_FILE = 'data/road_anomaly_metrics.json'


# SUMO_CMD = ['sumo', '-n', NET_FILE, '-r', ROUTE_FILE, '-a', ADDITIONAL_FILE, '--step-length', '1.0']
SUMO_CMD = ['sumo-gui', '-c', SUMOCFG_FILE ,'--step-length', '1.0']
# SUMO_CMD = ['sumo', '-c', SUMOCFG_FILE, '--step-length', '1.0']

DAMAGE_EDGE_IDS = ['-4001.0.00', '-4002.0.00', '-5004.0.00']
# DAMAGE_EDGE_IDS = ['-4001.0.00', '-4002.0.00']

BATCH_SIZE = 3600  # number of detections to process in one batch
RESOLUTION = 1.0  # resolution of the grid in meters
PRIOR = 0.1  # prior probability of occupancy
MARGIN = 5.0  # margin distance for extending the occupancy grid map
OVERLAP_STEPS = 20  # number of steps for the overlap during the generation of the grid, this fixes the gaps between
# center line of connected edges
P_TRUE = 0.90  # probability of detection
P_FALSE = 0.05  # probability of false alarm
GPS_SIGMA = 5.0  # GPS noise in meters
DECAY_RATE = 0.2  # decay rate for the occupancy grid
SMOOTHING_SIGMA = 1.0  # sigma for the gaussian smoothing
SIM_STEPS = 1000  # number of simulation steps
SPEED_THRESHOLD = 3.0  # speed threshold for detecting damage This also used in i





PROB_DICT = load_road_anomaly_metrics(ROAD_ANOMALY_DETECTION_FILE)


# -------------------

def simulate():
    damage_model = RoadDamage(NET_FILE, DAMAGE_EDGE_IDS, radius=3.0, damage_file=ADDITIONAL_FILE, probability_file=PROBABILITY_FILE)
    # save the damage model according to the scenario name
    damage_model.save('data/' + SCENARIO_NAME + '/damage_model.json')

    print(f"Loaded damage model: {damage_model}")
    for dmg in damage_model.all_damages():
        print(" ", dmg)

    # grid = OccupancyGrid(NET_FILE, RESOLUTION, PRIOR)
    sensor_model = VehicleSensor(PROB_DICT, GPS_SIGMA, None, damage_model)
    # fusion_eng = FusionEngine(grid, sensor_model)
    sumo = SumoInterface(SUMO_CMD)

    detections = []
    veh_pos_last = {}
    for step in range(SIM_STEPS):
        sumo.step()
        damage_detected = False
        for vid, (x, y) in sumo.get_vehicle_positions().items():
            if sumo.get_vehicle_type(vid) != 'PasVeh':
                continue
            # filter the vehilcles wiht low speed
            if traci.vehicle.getSpeed(vid) < SPEED_THRESHOLD:
                continue
            if vid not in veh_pos_last:
                veh_pos_last[vid] = (x, y)
            else:
                # Use the last true position of the vehicle
                last_x, last_y = veh_pos_last[vid]
                veh_pos_last[vid] = (x, y)  # Update the last position
                detection = sensor_model.detect_damage_travel_position(step, last_x, last_y, x, y)
                if detection.detected:
                    print(f"[step {step}] Vehicle {vid} detected damage at ({detection.x:.2f}, {detection.y:.2f}) with "
                      f"type '{detection.type}'")
                    detections.append(detection)
                    damage_detected = True
            veh_pos_last[vid] = (x, y)  # Store the last true position of the vehicle
        if not damage_detected:
            detections.append(Detection(0, 0, step, False, "na"))

    detection_file_name = 'data/' + SCENARIO_NAME + '/detection_logs_' + str(SIM_STEPS) + '.txt'
    with open(detection_file_name, 'w') as f:
        for d in detections:
            f.write(f"{d.step} {d.x} {d.y} {d.detected} {d.type}\n")

    sumo.close()


def analyze(detection_file_name):
    with open(detection_file_name, 'r') as f:
        detections = []
        for line in f:
            step, x, y, detected, road_anomaly_type  = line.strip().split()
            detections.append(Detection(float(x), float(y), int(step), detected == 'True', road_anomaly_type))

    sensor = VehicleSensor(PROB_DICT, GPS_SIGMA, None)
    grid = OccupancyGrid(NET_FILE, sensor, RESOLUTION, PRIOR, MARGIN, OVERLAP_STEPS, DECAY_RATE, SMOOTHING_SIGMA)
    X_MIN, X_MAX, Y_MIN, Y_MAX = grid.x_min, grid.x_max, grid.y_min, grid.y_max

    grid.batch_update(detection_file_name,  batch_size=BATCH_SIZE)
    probmap_dict = grid.gen_probability_map()


    # load the damage_model.txt
    damage_list = RoadDamage.read('data/' + SCENARIO_NAME + '/damage_model.json')

    damage_coords = gen_damage_area(damage_list, X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, plot_margin=20)



    # plot the probmap for each road anomaly type
    for anomaly_type in sensor.anomaly_types:
        # why is probmap empty
        # show the maximum probability of the probmap which is not nan
        probmap = probmap_dict[anomaly_type]
        probmap = np.nan_to_num(probmap, nan=0.0)  # replace NaN with 0
        print("Max probability in probmap for {} is: {}".format(anomaly_type, np.max(probmap_dict[anomaly_type])) )

        plt.figure(figsize=(8, 8))
        plt.imshow(probmap, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
        plt.colorbar(label='P(occupied)')
        plt.title('Road Damage Occupancy Grid Map (Final)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.savefig('image/occupancy_grid_' + anomaly_type + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        probmap_sliced = probmap[damage_coords['y_min_idx']:damage_coords['y_max_idx'], damage_coords['x_min_idx']:damage_coords['x_max_idx']]
        plt.figure(figsize=(8, 8))
        plt.imshow(probmap_sliced, origin='lower', extent=(damage_coords['x_min'], damage_coords['x_max'], damage_coords['y_min'], damage_coords['y_max']))
        plt.colorbar(label='P(occupied)')
        plt.title('Road Anomaly Occupancy Grid Map (Sliced)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.savefig('image/occupancy_grid_sliced_' + anomaly_type + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()





    # Convert detections to arrays
    x_coords = np.array([d.x for d in detections if d.detected])
    y_coords = np.array([d.y for d in detections if d.detected])

    # Create a 2D histogram
    bins = 100  # Adjust the number of bins for resolution
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins)

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap.T, origin='lower', extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap='Blues')
    plt.colorbar(label='Detection Density')
    plt.title('Vehicle Damage Detections (Binned)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.savefig('image/detection_density_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # simulate()
    analyze( detection_file_name = 'data/' + SCENARIO_NAME + '/detection_logs_' + str(SIM_STEPS) + '.txt')

