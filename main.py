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
from utilities import load_road_anomaly_metrics, gen_damage_area, visualize_clustered_map, compare_anomaly_results
import traci
import argparse
import json
import os
from pathlib import Path


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
                      f"type '{detection.type}' evaluated as {detection.eval}")
                    detections.append(detection)
                    damage_detected = True
            veh_pos_last[vid] = (x, y)  # Store the last true position of the vehicle
        if not damage_detected:
            detections.append(Detection(0, 0, step, False, "na"))

    detection_file_name = 'data/' + SCENARIO_NAME + '/detection_logs_' + str(SIM_STEPS) + '.txt'
    with open(detection_file_name, 'w') as f:
        for d in detections:
            f.write(f"{d.step} {d.x} {d.y} {d.detected} {d.type} {d.eval}\n")

    sumo.close()


def analyze(detection_file_name):
    # Create a copy of the default colormap and set the 'bad' (NaN) color

    with open(detection_file_name, 'r') as f:
        detections = []
        for line in f:
            step, x, y, detected, road_anomaly_type, evaluation  = line.strip().split()
            detections.append(Detection(float(x), float(y), int(step), detected == 'True', road_anomaly_type, evaluation))

    sensor = VehicleSensor(PROB_DICT, GPS_SIGMA, None)
    grid = OccupancyGrid(NET_FILE, sensor, RESOLUTION, None, PRIOR_MILD, MARGIN, OVERLAP_STEPS, DECAY_RATE, SMOOTHING_SIGMA, PROB_THRESHOLD, NEIGHBOR_DEPTH)
    X_MIN, X_MAX, Y_MIN, Y_MAX = grid.x_min, grid.x_max, grid.y_min, grid.y_max

    export_json_path = "data/" + SCENARIO_NAME + "/result_" + str(SIM_STEPS) + ".json"

    prob_map_dict, max_prob_map, filtered_prob_map, filtered_prob_map_type, clustered_prob_map, clustered_type_map, region_id_map = grid.gen_probability_map(detection_file_name,
                                                                                                                                                             batch_size=BATCH_SIZE,
                                                                                                                                                             average_neighbors=True,
                                                                                                                                                             export_json_path=export_json_path)

    # load the damage_model.txt
    damage_list = RoadDamage.read('data/' + SCENARIO_NAME + '/damage_model.json')
    damage_coords = gen_damage_area(damage_list, X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, plot_margin=20)

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
    plt.savefig('image/' + SCENARIO_NAME +  '/detection_density_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plot the probmap for each road anomaly type
    for anomaly_type in sensor.anomaly_types:
        # why is probmap empty
        # show the maximum probability of the probmap which is not nan
        probmap = prob_map_dict[anomaly_type]
        probmap = np.nan_to_num(probmap, nan=0.0)  # replace NaN with 0
        print("Max probability in probmap for {} is: {}".format(anomaly_type, np.max(prob_map_dict[anomaly_type])) )

        plt.figure(figsize=(8, 8))
        plt.imshow(probmap, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
        plt.colorbar(label='P(occupied)')
        plt.title('Road Damage Occupancy Grid Map (Final)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.savefig('image/' + SCENARIO_NAME +  '/occupancy_grid_' + anomaly_type + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # probmap_sliced = probmap[damage_coords['y_min_idx']:damage_coords['y_max_idx'], damage_coords['x_min_idx']:damage_coords['x_max_idx']]
        # plt.figure(figsize=(8, 8))
        # plt.imshow(probmap_sliced, origin='lower', extent=(damage_coords['x_min'], damage_coords['x_max'], damage_coords['y_min'], damage_coords['y_max']))
        # plt.colorbar(label='P(occupied)')
        # plt.title('Road Anomaly Occupancy Grid Map (Sliced)')
        # plt.xlabel('X [m]')
        # plt.ylabel('Y [m]')
        # plt.savefig('image/' + SCENARIO_NAME +  '/occupancy_grid_sliced_' + anomaly_type + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.close()

    # plot the maximum probability map
    plt.figure(figsize=(8, 8))
    plt.imshow(max_prob_map, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    plt.colorbar(label='Max P(occupied)')
    plt.title('Road Damage Maximum Probability Map (Final)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.savefig('image/' + SCENARIO_NAME +  '/max_probability_map_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # plot the filtered probability map
    plt.figure(figsize=(8, 8))
    plt.imshow(filtered_prob_map, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    plt.colorbar(label='Filtered P(occupied)')
    plt.title('Road Damage Filtered Probability Map (Final)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.savefig('image/' + SCENARIO_NAME +  '/filtered_probability_map_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # plot the clustered probability map and mark the region id based on the region_id_map. The region_id is put next to the cluster of road anomalies
    # plt.figure(figsize=(8, 8))
    # plt.imshow(clustered_prob_map, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    # plt.colorbar(label='Clustered P(occupied)')
    # plt.title('Road Damage Clustered Probability Map (Final)')
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')

    # ploted_id = set()  # to keep track of the region ids that have been plotted
    # for y in range(region_id_map.shape[0]):
    #     for x in range(region_id_map.shape[1]):
    #         if region_id_map[y, x] is not "na" and region_id_map[y, x] not in ploted_id:  # only plot the region id if it is greater than 0
    #             plt.text(x * RESOLUTION + X_MIN, y * RESOLUTION + Y_MIN, str(region_id_map[y, x]), color='red', fontsize=8,
    #                      ha='center', va='center')
    #             ploted_id.add(region_id_map[y, x])



    # plt.savefig('image/' + SCENARIO_NAME +  '/clustered_probability_map_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()

    visualize_clustered_map(clustered_prob_map, filtered_prob_map_type, region_id_map, save_path="image/" + SCENARIO_NAME + '/clustered_probability_map_' + str(SIM_STEPS) + '.png',)

    results = compare_anomaly_results(gt_json_path="data/" + SCENARIO_NAME + '/damage_model.json', det_json_path="data/" + SCENARIO_NAME + '/result_' + str(SIM_STEPS) + ".json",
                                      save_path="image/" + SCENARIO_NAME + "/compare_" + str(SIM_STEPS) + ".png", containment_threshold=0.5)
    print(results)









def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SUMO simulation with a given scenario configuration."
    )
    parser.add_argument(
        "--config_file",
        "-c",
        default="config/brussels_rural_config.json",
        help=(
            "Path to the JSON config file. "
            "If only a filename is given, it is assumed to be in the 'config/' folder.\n"
            "Example: --config_file brussels_rural_config.json"
        ),
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["simulate", "analyze", "both"],
        default="both",
        help=(
            "Choose what to run:\n"
            "'simulate' – only run the SUMO simulation\n"
            "'analyze' – only run the analysis step\n"
            "'both' – run simulation followed by analysis (default)"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve config path:
    # - if user passes just a filename, use config/<filename>
    # - if they pass a full/relative path, use as is
    if os.path.dirname(args.config_file):
        config_path = Path(args.config_file)
    else:
        config_path = Path("config") / args.config_file

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        params = json.load(f)

    print(f"Loaded parameters from {config_path}:")
    print(params)


    # Access variables
    SCENARIO_NAME = params["SCENARIO_NAME"]
    NET_FILE = params["NET_FILE"]
    ADDITIONAL_FILE = params["ADDITIONAL_FILE"]
    PROBABILITY_FILE = params["PROBABILITY_FILE"]
    ROAD_ANOMALY_DETECTION_FILE = params["ROAD_ANOMALY_DETECTION_FILE"]
    SUMOCFG_FILE = params["SUMOCFG_FILE"]
    ROUTE_FILE = params["ROUTE_FILE"]
    DAMAGE_EDGE_IDS = params["DAMAGE_EDGE_IDS"]

    SUMO_CMD = ["sumo", "-c", SUMOCFG_FILE, "--step-length", "1.0"]

    BATCH_SIZE = params["BATCH_SIZE"]
    RESOLUTION = params["RESOLUTION"]
    PRIOR_MILD = params["PRIOR_MILD"]
    MARGIN = params["MARGIN"]
    OVERLAP_STEPS = params["OVERLAP_STEPS"]
    GPS_SIGMA = params["GPS_SIGMA"]
    DECAY_RATE = params["DECAY_RATE"]
    SMOOTHING_SIGMA = params["SMOOTHING_SIGMA"]
    SIM_STEPS = params["SIM_STEPS"]
    SPEED_THRESHOLD = params["SPEED_THRESHOLD"]
    PROB_THRESHOLD = params["PROB_THRESHOLD"]
    NEIGHBOR_DEPTH = params["NEIGHBOR_DEPTH"]

    # Load anomaly metrics
    PROB_DICT = load_road_anomaly_metrics(ROAD_ANOMALY_DETECTION_FILE)

    if args.mode in ["simulate", "both"]:
        simulate()

    if args.mode in ["analyze", "both"]:
        detection_file = Path("data") / SCENARIO_NAME / f"detection_logs_{SIM_STEPS}.txt"
        analyze(detection_file_name=str(detection_file))


