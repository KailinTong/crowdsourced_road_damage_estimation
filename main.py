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
import sys
from pathlib import Path

import csv
from datetime import datetime

# NEW: 11.02.2026 DORA
#############################################################
# --- Experiment controls (used for file naming / sweeps) ---
FLOW_SCALE = 1.0  # 0.5, 1.0, 1.5 ...
EXP_TAG = ""      # set per run

def make_exp_tag(sim_steps: int, flow_scale: float, decay_rate: float) -> str:
    # Safe, readable tag for filenames
    return f"steps{sim_steps}_scale{flow_scale:g}_decay{decay_rate:g}"
#############################################################

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
    interrupted = False
    try:
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
                        if not SIM_QUIET:
                            print(f"[step {step}] Vehicle {vid} detected damage at ({detection.x:.2f}, {detection.y:.2f}) with "
                              f"type '{detection.type}' evaluated as {detection.eval}")
                        detections.append(detection)
                        damage_detected = True
                veh_pos_last[vid] = (x, y)  # Store the last true position of the vehicle
            if not damage_detected:
                detections.append(Detection(0, 0, step, False, "na"))
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Ctrl+C] Stopping. Saving detections and closing SUMO...")
    finally:
        # NEW: 11.02.2026 DORA
        #############################################################
        # detection_file_name = 'data/' + SCENARIO_NAME + '/detection_logs_' + str(SIM_STEPS) + '.txt'
        os.makedirs(f"data/{SCENARIO_NAME}", exist_ok=True)
        detection_file_name = f"data/{SCENARIO_NAME}/detection_logs_{EXP_TAG}.txt"
        #############################################################
        with open(detection_file_name, 'w') as f:
            for d in detections:
                f.write(f"{d.step} {d.x} {d.y} {d.detected} {d.type} {d.eval}\n")
        sumo.close()
    if interrupted:
        sys.exit(0)


def analyze(detection_file_name): 
    import matplotlib.pyplot as plt
    from grid import OccupancyGrid
    # Create a copy of the default colormap and set the 'bad' (NaN) color

    with open(detection_file_name, 'r') as f:
        detections = []
        for line in f:
            step, x, y, detected, road_anomaly_type, evaluation  = line.strip().split()
            detections.append(Detection(float(x), float(y), int(step), detected == 'True', road_anomaly_type, evaluation))

    sensor = VehicleSensor(PROB_DICT, GPS_SIGMA, None)

    # Load risk regions from PROBABILITY_FILE if available
    risk_regions = []
    risk_prior_by_level = {}
    try:
        import json
        from shapely.wkt import loads as load_wkt
        with open(PROBABILITY_FILE, 'r') as f:
            prob_data = json.load(f)
        
        # Determine unique probability levels
        unique_probs = sorted(list(set(item['probabilities']['vehicle'] for item in prob_data.values())))
        # value -> integer level
        prob_to_level = {p: i for i, p in enumerate(unique_probs)}
        # integer level -> value (for OccupancyGrid)
        risk_prior_by_level = {i: p for p, i in prob_to_level.items()}

        print(f"Loaded {len(prob_data)} risk zones. Risk Levels: {risk_prior_by_level}")

        for item in prob_data.values():
            poly = load_wkt(item['polygon'])
            prob = item['probabilities']['vehicle']
            level = prob_to_level[prob]
            risk_regions.append({'geometry': poly, 'level': level})

    except Exception as e:
        print(f"Warning: Could not load risk regions for OccupancyGrid: {e}")
        risk_regions = None
        risk_prior_by_level = None

    grid = OccupancyGrid(NET_FILE, sensor, RESOLUTION, None, PRIOR_MILD, MARGIN, OVERLAP_STEPS, DECAY_RATE, SMOOTHING_SIGMA, PROB_THRESHOLD, NEIGHBOR_DEPTH, 
                         risk_regions=risk_regions, 
                         risk_prior_by_level=risk_prior_by_level)
    X_MIN, X_MAX, Y_MIN, Y_MAX = grid.x_min, grid.x_max, grid.y_min, grid.y_max

    # NEW: 11.02.2026 DORA
    #############################################################
    # export_json_path = "data/" + SCENARIO_NAME + "/result_" + str(SIM_STEPS) + ".json"
    os.makedirs(f"data/{SCENARIO_NAME}", exist_ok=True)
    os.makedirs(f"image/{SCENARIO_NAME}", exist_ok=True)

    export_json_path = f"data/{SCENARIO_NAME}/result_{EXP_TAG}.json"
    #############################################################
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
    # NEW: 11.02.2026 DORA
    #############################################################
    # plt.savefig('image/' + SCENARIO_NAME +  '/detection_density_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(f"image/{SCENARIO_NAME}/detection_density_{EXP_TAG}.png", dpi=300, bbox_inches='tight')
    plt.show()
    #############################################################
    plt.close()

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
        # NEW: 11.02.2026 DORA
        #############################################################
        # plt.savefig('image/' + SCENARIO_NAME +  '/occupancy_grid_' + anomaly_type + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(f"image/{SCENARIO_NAME}/occupancy_grid_{anomaly_type}_{EXP_TAG}.png", dpi=300, bbox_inches='tight')
        #############################################################
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
    # NEW: 11.02.2026 DORA
    #############################################################
    # plt.savefig('image/' + SCENARIO_NAME +  '/max_probability_map_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(f"image/{SCENARIO_NAME}/max_probability_map_{EXP_TAG}.png", dpi=300, bbox_inches='tight')
    #############################################################
    plt.show()
    plt.close()

    # plot the filtered probability map
    plt.figure(figsize=(8, 8))
    plt.imshow(filtered_prob_map, origin='lower', extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    plt.colorbar(label='Filtered P(occupied)')
    plt.title('Road Damage Filtered Probability Map (Final)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    # NEW: 11.02.2026 DORA
    #############################################################
    # plt.savefig('image/' + SCENARIO_NAME +  '/filtered_probability_map_' + str(SIM_STEPS) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(f"image/{SCENARIO_NAME}/filtered_probability_map_{EXP_TAG}.png", dpi=300, bbox_inches='tight')
    #############################################################
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

    #  NEW: 11.02.2026 DORA
    #############################################################
    # visualize_clustered_map(clustered_prob_map, filtered_prob_map_type, region_id_map, save_path="image/" + SCENARIO_NAME + '/clustered_probability_map_' + str(SIM_STEPS) + '.png',)
    visualize_clustered_map(
        clustered_prob_map, filtered_prob_map_type, region_id_map,
        save_path=f"image/{SCENARIO_NAME}/clustered_probability_map_{EXP_TAG}.png"
    )
    #############################################################
    
    #  NEW: 11.02.2026 DORA
    #############################################################
    # results = compare_anomaly_results(gt_json_path="data/" + SCENARIO_NAME + '/damage_model.json', det_json_path="data/" + SCENARIO_NAME + '/result_' + str(SIM_STEPS) + ".json",
                                    #   save_path="image/" + SCENARIO_NAME + "/compare_" + str(SIM_STEPS) + ".png", containment_threshold=0.5)
    results = compare_anomaly_results(
    gt_json_path=f"data/{SCENARIO_NAME}/damage_model.json",
    det_json_path=f"data/{SCENARIO_NAME}/result_{EXP_TAG}.json",
    save_path=f"image/{SCENARIO_NAME}/compare_{EXP_TAG}.png",
    iou_threshold=0.5
    )
    #############################################################
    print(results)

# NEW: 11.02.2026 DORA
#############################################################
def run_convergence_study(sim_steps_list, flow_scales, decay_rates, *, mode="both"):
    global SIM_STEPS, DECAY_RATE, FLOW_SCALE, EXP_TAG, SUMO_CMD

    # Keep a base SUMO command without scale, then rebuild per run
    base_cmd = [c for c in SUMO_CMD if c != "--scale"]
    # If "--scale X" already exists, remove both tokens
    cleaned = []
    skip_next = False
    for tok in SUMO_CMD:
        if skip_next:
            skip_next = False
            continue
        if tok == "--scale":
            skip_next = True
            continue
        cleaned.append(tok)
    base_cmd = cleaned

    for scale in flow_scales:
        for steps in sim_steps_list:
            for decay in decay_rates:
                FLOW_SCALE = float(scale)
                SIM_STEPS = int(steps)
                DECAY_RATE = float(decay)

                EXP_TAG = make_exp_tag(SIM_STEPS, FLOW_SCALE, DECAY_RATE)
                print(f"\n=== Running EXP: {EXP_TAG} ===")

                # rebuild SUMO_CMD with scale
                SUMO_CMD = base_cmd.copy()
                SUMO_CMD.extend(["--scale", str(FLOW_SCALE)])

                if mode in ("simulate", "both"):
                    simulate()

                if mode in ("analyze", "both"):
                    det_file = f"data/{SCENARIO_NAME}/detection_logs_{EXP_TAG}.txt"
                    analyze(detection_file_name=det_file)
#############################################################

# NEW: 11.02.2026 DORA
#############################################################
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Run SUMO simulation with a given scenario configuration."
#     )
#     parser.add_argument(
#         "--config_file",
#         "-c",
#         default="config/brussels_rural_config.json",
#         help=(
#             "Path to the JSON config file. "
#             "If only a filename is given, it is assumed to be in the 'config/' folder.\n"
#             "Example: --config_file brussels_rural_config.json"
#         ),
#     )

#     parser.add_argument(
#         "--mode",
#         "-m",
#         choices=["simulate", "analyze", "both"],
#         default="both",
#         help=(
#             "Choose what to run:\n"
#             "'simulate' – only run the SUMO simulation\n"
#             "'analyze' – only run the analysis step\n"
#             "'both' – run simulation followed by analysis (default)"
#         ),
#     )

#     parser.add_argument(
#         "--gui",
#         action="store_true",
#         help="Run with SUMO GUI instead of headless.",
#     )
#     parser.add_argument(
#         "--quiet", "-q",
#         action="store_true",
#         help="Suppress per-detection step logs during simulation (detections still saved to file).",
#     )
#     parser.add_argument(
#         "--scale",
#         type=float,
#         default=None,
#         help="Scale SUMO traffic flow (e.g. 0.5=half, 2.0=double). Used by run_timeseries_study for flow sensitivity.",
#     )

#     args = parser.parse_args()

#     # Resolve config path:
#     # - if user passes just a filename, use config/<filename>
#     # - if they pass a full/relative path, use as is
#     if os.path.dirname(args.config_file):
#         config_path = Path(args.config_file)
#     else:
#         config_path = Path("config") / args.config_file

#     if not config_path.is_file():
#         raise FileNotFoundError(f"Config file not found: {config_path}")

#     with config_path.open("r") as f:
#         params = json.load(f)

#     print(f"Loaded parameters from {config_path}:")
#     print(params)

#     # Assign to globals so simulate() and analyze() can use them
#     global SCENARIO_NAME, NET_FILE, ADDITIONAL_FILE, PROBABILITY_FILE, DAMAGE_EDGE_IDS, SUMO_CMD
#     global BATCH_SIZE, RESOLUTION, PRIOR_MILD, MARGIN, OVERLAP_STEPS, GPS_SIGMA, DECAY_RATE
#     global SMOOTHING_SIGMA, SIM_STEPS, SPEED_THRESHOLD, PROB_THRESHOLD, NEIGHBOR_DEPTH, PROB_DICT, SIM_QUIET

#     # Access variables
#     SCENARIO_NAME = params["SCENARIO_NAME"]
#     NET_FILE = params["NET_FILE"]
#     ADDITIONAL_FILE = params["ADDITIONAL_FILE"]
#     PROBABILITY_FILE = params["PROBABILITY_FILE"]
#     ROAD_ANOMALY_DETECTION_FILE = params["ROAD_ANOMALY_DETECTION_FILE"]
#     SUMOCFG_FILE = params["SUMOCFG_FILE"]
#     ROUTE_FILE = params["ROUTE_FILE"]
#     DAMAGE_EDGE_IDS = params["DAMAGE_EDGE_IDS"]

#     sumo_binary = "sumo-gui" if args.gui else "sumo"

#     # Resolve sumo/sumo-gui path via SUMO_HOME or common Windows install path
#     if "/" not in sumo_binary and "\\" not in sumo_binary:
#         sumo_home = os.environ.get("SUMO_HOME")
#         if not sumo_home and sys.platform == "win32":
#             for cand in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"]:
#                 if os.path.isdir(os.path.join(cand, "bin")):
#                     sumo_home = cand
#                     break
#         if sumo_home:
#             if sys.platform == "win32":
#                 exe = "sumo-gui.exe" if args.gui else "sumo.exe"
#             else:
#                 exe = "sumo-gui" if args.gui else "sumo"
#             resolved = os.path.join(sumo_home, "bin", exe)
#             if os.path.isfile(resolved):
#                 sumo_binary = resolved
#             elif args.gui:
#                 print(f"提示: 在 SUMO_HOME/bin 下未找到 {exe}，请检查 SUMO 安装。将尝试使用 PATH 中的 'sumo-gui'。")
#         elif args.gui:
#             print("提示: 未设置 SUMO_HOME。请确保 'sumo-gui' 在系统 PATH 中，或将 SUMO_HOME 设为 SUMO 安装目录（如 C:\\Program Files (x86)\\Eclipse SUMO）。")
#     SUMO_CMD = [sumo_binary, "-c", SUMOCFG_FILE, "--step-length", "1.0"]
#     if args.scale is not None:
#         SUMO_CMD.extend(["--scale", str(args.scale)])

#     BATCH_SIZE = params["BATCH_SIZE"]
#     RESOLUTION = params["RESOLUTION"]
#     PRIOR_MILD = params["PRIOR_MILD"]
#     MARGIN = params["MARGIN"]
#     OVERLAP_STEPS = params["OVERLAP_STEPS"]
#     GPS_SIGMA = params["GPS_SIGMA"]
#     DECAY_RATE = params["DECAY_RATE"]
#     SMOOTHING_SIGMA = params["SMOOTHING_SIGMA"]
#     SIM_STEPS = params["SIM_STEPS"]
#     SPEED_THRESHOLD = params["SPEED_THRESHOLD"]
#     PROB_THRESHOLD = params["PROB_THRESHOLD"]
#     NEIGHBOR_DEPTH = params["NEIGHBOR_DEPTH"]

#     # Load anomaly metrics
#     PROB_DICT = load_road_anomaly_metrics(ROAD_ANOMALY_DETECTION_FILE)
#     SIM_QUIET = args.quiet

#     # NEW: 11.02.2026 DORA
#     #############################################################
#     # if args.mode in ["simulate", "both"]:
#     #     simulate()

#     # if args.mode in ["analyze", "both"]:
#     #     detection_file = Path("data") / SCENARIO_NAME / f"detection_logs_{SIM_STEPS}.txt"
#     #     analyze(detection_file_name=str(detection_file))
#     return args
#     #############################################################
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

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI instead of headless.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-detection step logs during simulation (detections still saved to file).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale SUMO traffic flow (e.g. 0.5=half, 2.0=double).",
    )

    return parser.parse_args()

def load_and_set_globals(args):
    """
    Load scenario config JSON, set module-level globals used by simulate() and analyze(),
    and build SUMO_CMD. Does NOT run simulate()/analyze().
    """
    # Assign to globals so simulate() and analyze() can use them
    global SCENARIO_NAME, NET_FILE, ADDITIONAL_FILE, PROBABILITY_FILE, DAMAGE_EDGE_IDS, SUMO_CMD
    global BATCH_SIZE, RESOLUTION, PRIOR_MILD, MARGIN, OVERLAP_STEPS, GPS_SIGMA, DECAY_RATE
    global SMOOTHING_SIGMA, SIM_STEPS, SPEED_THRESHOLD, PROB_THRESHOLD, NEIGHBOR_DEPTH, PROB_DICT, SIM_QUIET
    global ROAD_ANOMALY_DETECTION_FILE

    # Resolve config path:
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

    # Build SUMO command
    sumo_binary = "sumo-gui" if getattr(args, "gui", False) else "sumo"

    # Resolve sumo/sumo-gui path via SUMO_HOME or common Windows install path
    if "/" not in sumo_binary and "\\" not in sumo_binary:
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home and sys.platform == "win32":
            for cand in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"]:
                if os.path.isdir(os.path.join(cand, "bin")) and os.path.isdir(cand):
                    sumo_home = cand
                    break
        if sumo_home:
            if sys.platform == "win32":
                exe = "sumo-gui.exe" if getattr(args, "gui", False) else "sumo.exe"
            else:
                exe = "sumo-gui" if getattr(args, "gui", False) else "sumo"
            resolved = os.path.join(sumo_home, "bin", exe)
            if os.path.isfile(resolved):
                sumo_binary = resolved
            elif getattr(args, "gui", False):
                print(f"提示: 在 SUMO_HOME/bin 下未找到 {exe}，请检查 SUMO 安装。将尝试使用 PATH 中的 'sumo-gui'。")
        elif getattr(args, "gui", False):
            print("提示: 未设置 SUMO_HOME。请确保 'sumo-gui' 在系统 PATH 中，或将 SUMO_HOME 设为 SUMO 安装目录。")

    SUMO_CMD = [sumo_binary, "-c", SUMOCFG_FILE, "--step-length", "1.0"]
    if getattr(args, "scale", None) is not None:
        SUMO_CMD.extend(["--scale", str(args.scale)])

    # Numerical / model parameters
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

    # Load anomaly metrics (FIX: must call load_road_anomaly_metrics)
    PROB_DICT = load_road_anomaly_metrics(ROAD_ANOMALY_DETECTION_FILE)

    # Quiet flag
    SIM_QUIET = getattr(args, "quiet", False)

#############################################################

# NEW: 11.02.2026 DORA
#############################################################
def main():
    args = parse_args()
    load_and_set_globals(args)

    global EXP_TAG, FLOW_SCALE
    FLOW_SCALE = args.scale if args.scale is not None else 1.0
    EXP_TAG = make_exp_tag(SIM_STEPS, FLOW_SCALE, DECAY_RATE)

    if args.mode in ("simulate", "both"):
        simulate()
    if args.mode in ("analyze", "both"):
        det_file = f"data/{SCENARIO_NAME}/detection_logs_{EXP_TAG}.txt"
        analyze(detection_file_name=det_file)



# if __name__ == "__main__":
#     # ======= Control your experiment here in Cursor =======
#     # You can run your sweep by clicking Run.

#     class SimpleArgs:
#         config_file = "config/Graz_A2_config.json" # Graz_A2_config.json/brussels_rural_config.json
#         mode = "both"
#         gui = False
#         quiet = False
#         scale = None

#     args = SimpleArgs()
#     # reuse the same loader
#     load_and_set_globals(args)

#     # --- Your sweep parameters ---
#     SIM_STEPS_LIST = [9000]#300, 450, 900, 1800, 3600, 5400, 7200, 
#     FLOW_SCALES = [1.0]#0.5, 1.0, 1.5
#     DECAY_RATES = [0.01, 0.05, 0.1]

#     # run the full factorial study
#     run_convergence_study(SIM_STEPS_LIST, FLOW_SCALES, DECAY_RATES, mode="both")

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # ======= Control your experiment here in Cursor =======
    # You can run your sweep by clicking Run.

    class SimpleArgs:
        config_file = "config/brussels_rural_config.json" # Graz_A2_config.json/brussels_rural_config.json
        mode = "both"
        gui = False
        quiet = False
        scale = None

    args = SimpleArgs()
    # reuse the same loader
    load_and_set_globals(args)

    # --- Your sweep parameters ---
    SIM_STEPS_LIST = [300, 450, 900, 1800, 3600, 5400, 7200, 9000]
    FLOW_SCALES = [1]
    DECAY_RATES = [0.15, 0.2]#0.01, 0.05, 0.1

    # run the full factorial study
    run_convergence_study(SIM_STEPS_LIST, FLOW_SCALES, DECAY_RATES, mode="both")