from typing import Dict
import json
from shapely.wkt import loads as load_wkt

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


import numpy as np
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




def compare_anomaly_results(gt_json_path: str, det_json_path: str, save_path: str, containment_threshold: float)-> dict:
    """
    Compare detected road anomalies against ground truth using containment/overlap.
    Also plots both sets for visual inspection.

    Matching rule:
        A detection matches a GT if they share the same anomaly type AND
        (GT area in detection / GT area) >= containment_threshold.

    Args:
        gt_json_path (str): Ground truth JSON.
        det_json_path (str): Detection result JSON.
        containment_threshold (float): Min fraction of GT covered by detection.

    Returns:
        dict: KPIs and match list.
    """
    # Load JSON files
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(det_json_path, "r") as f:
        det_data = json.load(f)

    # Parse polygons
    gt_polys = [(g["road_anomaly_type"], g["severity"], load_wkt(g["shape"])) for g in gt_data]
    det_polys = [(d["road_anomaly_type"], d["severity"], load_wkt(d["shape"])) for d in det_data]

    matches = []
    used_gt = set()
    used_det = set()

    # Matching: same type + containment/overlap
    for det_idx, (det_type, det_sev, det_poly) in enumerate(det_polys):
        best_cover = 0
        best_gt_idx = None
        for gt_idx, (gt_type, gt_sev, gt_poly) in enumerate(gt_polys):
            if gt_idx in used_gt:
                continue
            if gt_type != det_type:
                continue
            if not det_poly.is_valid or not gt_poly.is_valid:
                continue

            if det_poly.intersects(gt_poly):
                cover_ratio = det_poly.intersection(gt_poly).area / gt_poly.area
                if cover_ratio > best_cover:
                    best_cover = cover_ratio
                    best_gt_idx = gt_idx

        if best_gt_idx is not None and best_cover >= containment_threshold:
            matches.append({
                "det_idx": det_idx,
                "gt_idx": best_gt_idx,
                "cover_ratio": best_cover,
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
    mean_cover = sum(m["cover_ratio"] for m in matches) / TP if TP > 0 else 0

    results = {
        "overall": {
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_cover_ratio": mean_cover
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
        mean_cover_s = (sum(m["cover_ratio"] for m in matches if m["severity"] == sev) / TP_s) if TP_s > 0 else 0

        results["per_severity"][sev] = {
            "true_positives": TP_s,
            "false_positives": FP_s,
            "false_negatives": FN_s,
            "precision": prec_s,
            "recall": rec_s,
            "f1_score": f1_s,
            "mean_cover_ratio": mean_cover_s
        }

    # Plot for visual verification
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, _, poly in gt_polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue", label="Ground Truth" if "Ground Truth" not in ax.get_legend_handles_labels()[1] else "")
    for _, _, poly in det_polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="red", linestyle="--", label="Detection" if "Detection" not in ax.get_legend_handles_labels()[1] else "")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title("Ground Truth (blue) vs Detection (red)")
    # save the image in the same folder
    plt.savefig(save_path)
    plt.show()
    plt.close()

    return results


