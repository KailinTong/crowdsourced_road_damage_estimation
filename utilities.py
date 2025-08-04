from typing import Dict
import json
import numpy as np

# laod the probability dictionary from the json file
def load_road_anomaly_metrics(path: str = "data/road_anomaly_metrics.json") -> Dict[str, Dict[str, float]]:
    """
    Load road anomaly metrics from a JSON file into a Python dict.

    Parameters:
        path (str): Path to the JSON file containing the metrics.

    Returns:
        Dict[str, Dict[str, float]]:
            A dict where each key is an anomaly type (plus severity suffix if any),
            and each value is a dict with keys "tp", "fn", "fp" mapping to floats.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON metrics file: {e}")

    return data

def gen_damage_area(damage_list, X_MIN, X_MAX, Y_MIN, Y_MAX, RESOLUTION, plot_margin=20):

    damage_area = []
    for damage in damage_list:
        damage_area.append(damage.shape.bounds)
    damage_area = np.array(damage_area)
    damage_area = damage_area.reshape(-1, 4)
    damage_area = np.unique(damage_area, axis=0)
    # get the min and max of the damage area
    damage_x_min = np.min(damage_area[:, 0]) - plot_margin
    damage_x_max = np.max(damage_area[:, 2]) + plot_margin
    damage_y_min = np.min(damage_area[:, 1]) - plot_margin
    damage_y_max = np.max(damage_area[:, 3]) + plot_margin
    # slice the probmap only within the damage area, first convert the damage area to the probmap indices
    damage_x_min_idx = int((damage_x_min - X_MIN) / RESOLUTION)
    damage_x_max_idx = int((damage_x_max - X_MIN) / RESOLUTION)
    damage_y_min_idx = int((damage_y_min - Y_MIN) / RESOLUTION)
    damage_y_max_idx = int((damage_y_max - Y_MIN) / RESOLUTION)

    # convert the returns to a ditionary
    damage_coords = {
        "x_min": damage_x_min,
        "x_max": damage_x_max,
        "y_min": damage_y_min,
        "y_max": damage_y_max,
        "x_min_idx": damage_x_min_idx,
        "x_max_idx": damage_x_max_idx,
        "y_min_idx": damage_y_min_idx,
        "y_max_idx": damage_y_max_idx
    }
    return damage_coords


