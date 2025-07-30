from typing import Dict
import json

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