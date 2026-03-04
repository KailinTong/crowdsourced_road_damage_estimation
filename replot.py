import os
import json
import matplotlib.pyplot as plt
from shapely.wkt import loads as load_wkt
from utilities import compare_anomaly_results

# =========================
# USER CONFIG
# =========================
SCENARIO_NAME = "Graz_A2"   #brussels_rural/Graz_A2
FLOW_SCALE = 1
DECAY_RATE = 0.01
SIM_STEPS_LIST = [300, 1800, 5400, 9000]

IOU_THRESHOLD = 0.5

# =========================
# Custom replot function
# =========================

def replot_compare(gt_json_path, det_json_path, save_path):
    """
    Same as compare_anomaly_results but legend moved to lower right.
    """

    with open(gt_json_path, 'r') as f:
        gt_raw = json.load(f)

    if isinstance(gt_raw, dict):
        gt_data = list(gt_raw.values())
    else:
        gt_data = gt_raw

    with open(det_json_path, 'r') as f:
        det_data = json.load(f)

    gt_polys = []
    for g in gt_data:
        wkt = g.get("shape") or g.get("polygon")
        if not wkt:
            continue
        gt_polys.append(load_wkt(wkt))

    det_polys = []
    for d in det_data:
        wkt = d.get("shape") or d.get("polygon")
        if not wkt:
            continue
        det_polys.append(load_wkt(wkt))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot GT
    for poly in gt_polys:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.fill(x, y, color="blue", alpha=0.3, label="Ground Truth" 
                    if "Ground Truth" not in ax.get_legend_handles_labels()[1] else "")
            ax.plot(x, y, color="blue", linewidth=1)

    # Plot Detection
    for poly in det_polys:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linestyle="--", linewidth=1.5,
                    label="Detection" 
                    if "Detection" not in ax.get_legend_handles_labels()[1] else "")

    ax.set_aspect("equal", adjustable="box")

    # Move legend to LOWER RIGHT
    ax.legend(loc='lower right')

    ax.set_title(f"Comparison: steps={steps}, scale={FLOW_SCALE}, decay={DECAY_RATE}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# Run for selected configs
# =========================

for steps in SIM_STEPS_LIST:

    exp_tag = f"steps{steps}_scale{FLOW_SCALE}_decay{DECAY_RATE}"

    gt_json = f"data/{SCENARIO_NAME}/damage_model.json"
    det_json = f"data/{SCENARIO_NAME}/result_{exp_tag}.json"
    save_path = f"image/{SCENARIO_NAME}/compare_{exp_tag}_replot.png"

    if not os.path.exists(det_json):
        print(f"⚠ Skipping {steps} (file not found)")
        continue

    print(f"Replotting: {exp_tag}")
    replot_compare(gt_json, det_json, save_path)

print("Done.")