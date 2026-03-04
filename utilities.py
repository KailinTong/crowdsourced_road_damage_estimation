from typing import Dict, List
import json
import os
from shapely.wkt import loads as load_wkt
from shapely.geometry import shape, Polygon, MultiPolygon
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend

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


    return regions


def load_insar_risk_regions(geojson_path: str, risk_property: str = "risk_level") -> List[dict]:
    """
    Load InSAR-derived risk regions from a GeoJSON file.

    Each feature must have a valid geometry. Risk intensity is read from the
    specified property if present; otherwise, regions default to level 0.

    Returns:
        List[dict]: [{"geometry": shapely geometry, "risk_level": value, "level": value}]
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    regions = []
    for feature in data.get("features", []):
        geom_data = feature.get("geometry") or feature.get("Geometry")
        if geom_data is None:
            continue
        try:
            geom = shape(geom_data)
        except Exception:
            continue

        props = feature.get("properties", {}) or {}
        risk_value = props.get(risk_property)
        level = props.get("level", props.get("risk_class"))

        regions.append({
            "geometry": geom,
            "risk_level": risk_value,
            "level": level if level is not None else 0,
        })

    return regions


def parse_lanes(net_file: str) -> List[List[tuple]]:
    """
    Extract network bounds and lane shapes from a SUMO net file.
    Reused from scenario-specific visualization scripts.
    """
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        lanes = []
        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                shape_str = lane.attrib.get("shape", "")
                if not shape_str:
                    continue
                pts = []
                for pair in shape_str.strip().split():
                    coords = list(map(float, pair.split(",")))
                    pts.append((coords[0], coords[1]))
                lanes.append(pts)
        return lanes
    except Exception as e:
        print(f"Warning: Could not parse lanes from {net_file}: {e}")
        return []


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


import numpy as np
import matplotlib
# Use a non-interactive backend to avoid display errors during simulation if needed
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

def visualize_clustered_map(
    clustered_prob_map: np.ndarray,
    clustered_type_map: np.ndarray,
    region_id_map: np.ndarray,
    *,
    title: str = "Clustered Probability Map",
    save_path = "/",
    mild_road_color=(0.0, 0.0, 0.5),      # very dark blue
    anomaly_colors=None,                  # list of fixed colors for anomaly types
    use_alpha_by_prob: bool = True,
    prob_alpha_range: tuple[float, float] = (0.35, 1.0),
    mild_road_alpha: float = 0.85,
    y_up: bool = True,
    figsize=(10, 10),
    prior: float | None = None,
    label_mild_road: bool = False,
    max_labels: int | None = None,
    label_x_offset: float = 3.0
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    if anomaly_colors is None:
        # Up to 6 fixed anomaly colors (RGB tuples)
        anomaly_colors = [
            (1.0, 0.0, 0.0),   # red
            (1.0, 0.5, 0.0),   # orange
            (1.0, 1.0, 0.0),   # yellow
            (0.0, 1.0, 0.0),   # green
            (0.0, 1.0, 1.0),   # cyan
            (1.0, 0.0, 1.0)    # magenta
        ]

    H, W = clustered_prob_map.shape
    assert clustered_type_map.shape == (H, W) and region_id_map.shape == (H, W)

    # ---- Assign fixed colors ----
    unique_types = [t for t in np.unique(clustered_type_map) if t not in ("na", "", None)]
    type_colors = {}
    anomaly_idx = 0
    for t in unique_types:
        if t == "mild_road":
            type_colors[t] = mild_road_color
        else:
            type_colors[t] = anomaly_colors[anomaly_idx % len(anomaly_colors)]
            anomaly_idx += 1

    # ---- RGBA image ----
    rgba = np.zeros((H, W, 4), dtype=float)
    finite_probs = clustered_prob_map[np.isfinite(clustered_prob_map)]
    if prior is None and finite_probs.size > 0:
        p_min = float(np.nanmin(finite_probs))
        p_max = float(np.nanmax(finite_probs))
    else:
        p_min = float(prior if prior is not None else 0.0)
        p_max = float(np.nanmax(finite_probs)) if finite_probs.size > 0 else (prior if prior is not None else 1.0)

    def prob_to_alpha(p, is_mild: bool):
        if not use_alpha_by_prob:
            return mild_road_alpha if is_mild else prob_alpha_range[1]
        if is_mild:
            return mild_road_alpha
        if not np.isfinite(p):
            return 0.0
        if p_max <= p_min:
            return prob_alpha_range[1]
        x = (p - p_min) / (p_max - p_min)
        return float(np.clip(prob_alpha_range[0] + x * (prob_alpha_range[1] - prob_alpha_range[0]), 0.0, 1.0))

    for t, color in type_colors.items():
        mask = (clustered_type_map == t) & np.isfinite(clustered_prob_map)
        if not np.any(mask):
            continue
        rgba[mask, :3] = color
        alphas = np.vectorize(prob_to_alpha)(clustered_prob_map, t == "mild_road")
        rgba[mask, 3] = alphas[mask]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, origin=("lower" if y_up else "upper"), interpolation="none")

    # ---- Labels ----
    unique_rids = np.unique(region_id_map)
    candidates = []
    for rid in unique_rids:
        if rid in ("na", "", None):
            continue
        rmask = (region_id_map == rid)
        if not np.any(rmask):
            continue
        rtype = clustered_type_map[rmask][0]
        if (rtype == "mild_road") and not label_mild_road:
            continue
        rprob = float(clustered_prob_map[rmask][0])
        candidates.append((rid, rtype, rprob))

    candidates.sort(key=lambda x: x[2], reverse=True)
    if max_labels is not None:
        candidates = candidates[:max_labels]

    for rid, rtype, rprob in candidates:
        rmask = (region_id_map == rid)
        rr, cc = np.where(rmask)
        cy, cx = float(np.median(rr)), float(np.median(cc))
        cx += label_x_offset
        txt = ax.text(
            cx, cy, f"{rid} (p={rprob:.2f})",
            ha="left", va="center", fontsize=9, color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    # ---- Legend ----
    patches = [mpatches.Patch(color=color, label=t) for t, color in type_colors.items()]
    if patches:
        ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  framealpha=0.9, title="Type")

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()




def compare_anomaly_results(gt_json_path: str, det_json_path: str, save_path: str, iou_threshold: float = 0.3, net_file: str = None)-> dict:
    """
    Compare detected road anomalies against ground truth using IoU (Intersection over Union).
    Also plots both sets for visual inspection, optionally overlaid on the road network.

    Matching rule:
        A detection matches a GT if IoU >= iou_threshold.

    Args:
        gt_json_path (str): Ground truth JSON (can be damage_model.json or probabilities.json).
        det_json_path (str): Detection result JSON (result_*.json).
        save_path (str): Path to save the comparison plot.
        iou_threshold (float): Minimum IoU for a match.
        net_file (str): Optional path to SUMO network file for background.

    Returns:
        dict: KPIs and match list.
    """
    import matplotlib.pyplot as plt
    # Load JSON files
    with open(gt_json_path, 'r') as f:
        gt_raw = json.load(f)
    
    # Handle dict vs list for GT
    if isinstance(gt_raw, dict):
        gt_data = list(gt_raw.values())
    else:
        gt_data = gt_raw
        
    with open(det_json_path, 'r') as f:
        det_data = json.load(f)

    # Conversion logic with key mapping
    gt_polys = []
    for g in gt_data:
        # Try different shape keys
        wkt = g.get("shape") or g.get("polygon")
        if not wkt: continue
        
        # Try different type/severity keys
        type_str = g.get("road_anomaly_type") or "pothole"
        sev_str = g.get("severity") or "unknown"
        gt_polys.append((type_str, sev_str, load_wkt(wkt)))

    det_polys = []
    for d in det_data:
        wkt = d.get("shape") or d.get("polygon")
        if not wkt: continue
        type_str = d.get("road_anomaly_type") or "pothole"
        sev_str = d.get("severity") or "unknown"
        det_polys.append((type_str, sev_str, load_wkt(wkt)))

    matches = []
    used_gt = set()
    used_det = set()

    # Matching: IoU-based greedy matching
    for det_idx, (det_type, det_sev, det_poly) in enumerate(det_polys):
        best_iou = 0
        best_gt_idx = None
        for gt_idx, (gt_type, gt_sev, gt_poly) in enumerate(gt_polys):
            if gt_idx in used_gt:
                continue
            if not det_poly.is_valid or not gt_poly.is_valid:
                continue

            if det_poly.intersects(gt_poly):
                intersection = det_poly.intersection(gt_poly).area
                union = det_poly.union(gt_poly).area
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matches.append({
                "det_idx": det_idx,
                "gt_idx": best_gt_idx,
                "iou": best_iou,
                "severity": det_sev
            })
            used_gt.add(best_gt_idx)
            used_det.add(det_idx)

    # KPIs
    TP = len(matches)
    FP = len(det_polys) - TP
    FN = len(gt_polys) - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = sum(m["iou"] for m in matches) / TP if TP > 0 else 0

    results = {
        "overall": {
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_iou": avg_iou
        },
        "matches": matches,
        "per_severity": {}
    }

    # Per-severity stats
    severities = {"l", "m", "h"}
    for sev in severities:
        gt_sev = [i for i, (_, s, _) in enumerate(gt_polys) if s == sev]
        det_sev = [i for i, (_, s, _) in enumerate(det_polys) if s == sev]
        TP_s = sum(1 for m in matches if m["severity"] == sev)
        FP_s = len(det_sev) - TP_s
        FN_s = len(gt_sev) - TP_s
        prec_s = TP_s / (TP_s + FP_s) if TP_s + FP_s > 0 else 0
        rec_s = TP_s / (TP_s + FN_s) if TP_s + FN_s > 0 else 0
        f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
        avg_iou_s = (sum(m["iou"] for m in matches if m["severity"] == sev) / TP_s) if TP_s > 0 else 0

        results["per_severity"][sev] = {
            "true_positives": TP_s,
            "false_positives": FP_s,
            "false_negatives": FN_s,
            "precision": prec_s,
            "recall": rec_s,
            "f1_score": f1_s,
            "avg_iou": avg_iou_s
        }

    # Plot for visual verification
    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Plot Network Lanes first
    if net_file and os.path.exists(net_file):
        lanes = parse_lanes(net_file)
        for pts in lanes:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.3, zorder=0)

    # 2. Plot GT Polygons
    for _, _, poly in gt_polys:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.fill(x, y, color="blue", alpha=0.3, label="Ground Truth" if "Ground Truth" not in ax.get_legend_handles_labels()[1] else "", zorder=1)
            ax.plot(x, y, color="blue", linewidth=1, zorder=2)
        elif poly.geom_type == 'MultiPolygon':
            for sp in poly.geoms:
                x, y = sp.exterior.xy
                ax.fill(x, y, color="blue", alpha=0.3, zorder=1)
                ax.plot(x, y, color="blue", linewidth=1, zorder=2)

    # 3. Plot Detection Polygons
    for _, _, poly in det_polys:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linestyle="--", linewidth=1.5, label="Detection" if "Detection" not in ax.get_legend_handles_labels()[1] else "", zorder=3)
        elif poly.geom_type == 'MultiPolygon':
            for sp in poly.geoms:
                x, y = sp.exterior.xy
                ax.plot(x, y, color="red", linestyle="--", linewidth=1.5, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc='upper right')
    ax.set_title(f"Road Damage Comparison: GT (blue) vs Detection (red)\nMetrics: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return results


