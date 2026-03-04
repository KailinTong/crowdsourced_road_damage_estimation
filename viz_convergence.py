# viz_convergence.py
# Post-process existing runs to visualize convergence vs SIM_STEPS, flow_scale, decay_rate.
#
# Usage examples:
#   python viz_convergence.py --scenario brussels_rural
#   python viz_convergence.py --scenario brussels_rural --iou 0.5
#   python viz_convergence.py --scenario brussels_rural --skip_compare_plots
#
# Assumptions:
#   - data/<SCENARIO_NAME>/damage_model.json exists (ground truth)
#   - data/<SCENARIO_NAME>/result_<EXP_TAG>.json exists for each run
#   - EXP_TAG format: steps{SIM}_scale{FLOW}_decay{DECAY}
#
# Outputs:
#   - data/<SCENARIO_NAME>/convergence_metrics.csv
#   - image/<SCENARIO_NAME>/convergence_summary/*.png

import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import json
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon



from shapely import wkt
from shapely.errors import GEOSException
try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None

def _clean_geom(g):
    """
    Repair invalid geometries to avoid GEOS TopologyException.
    Returns a geometry that is valid or None if cannot be repaired.
    """
    if g is None or g.is_empty:
        return None

    # Fast path
    if getattr(g, "is_valid", True):
        return g

    # Try Shapely 2.x make_valid
    if make_valid is not None:
        try:
            g2 = make_valid(g)
            if not g2.is_empty:
                return g2
        except Exception:
            pass

    # Fallback: classic trick for self-intersections
    try:
        g2 = g.buffer(0)
        if not g2.is_empty:
            return g2
    except Exception:
        return None

    return None


def load_objects_from_json(json_path: str):
    """
    Load GT or detection objects from JSON.

    Expected JSON structure: a list of dicts, each containing:
      - "id"
      - "shape": WKT string like "POLYGON ((...))"
      - optional: "severity", "road_anomaly_type", "probability", "centroid", ...

    Returns:
      objs: list of dicts with keys:
        - id (str)
        - geom (shapely geometry)
        - severity (str or None)
        - road_anomaly_type (str or None)
        - probability (float or None)
        - centroid (tuple[float,float] or None)
        - raw (original dict)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a list of objects, got: {type(data)}")

    objs = []
    for item in data:
        if "shape" not in item:
            # Skip or raise depending on your preference
            # raise ValueError(f"Missing 'shape' field in item: {item}")
            continue

        geom = wkt.loads(item["shape"])


        geom = _clean_geom(geom)
        if geom is None:
            continue


        # Normalize to polygon-like geometry (Polygon or MultiPolygon)
        if geom.is_empty:
            continue

        # Sometimes WKT parsing can yield GeometryCollection; keep only polygons if so
        gtype = geom.geom_type
        if gtype == "Polygon":
            poly = geom
        elif gtype == "MultiPolygon":
            poly = geom
        else:
            # Try to extract polygonal parts if possible
            try:
                polys = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
                if not polys:
                    continue
                # Merge list into MultiPolygon-like (keep first if you want)
                poly = polys[0]
            except Exception:
                continue

        centroid = None
        c = item.get("centroid", None)
        if isinstance(c, (list, tuple)) and len(c) == 2:
            centroid = (float(c[0]), float(c[1]))

        prob = item.get("probability", None)
        if prob is not None:
            try:
                prob = float(prob)
            except Exception:
                prob = None

        objs.append({
            "id": item.get("id", None),
            "geom": poly,
            "severity": item.get("severity", None),
            "road_anomaly_type": item.get("road_anomaly_type", None),
            "probability": prob,
            "centroid": centroid,
            "raw": item,
        })

    return objs

def compute_coverage_metrics(
    gt_objs,
    det_objs,
    *,
    iog_thresholds=(0.5, 0.8, 0.95),
    by_severity=True,
    by_type=True
):
    """
    Compute coverage-style convergence metrics between GT and detections.

    Key concept:
      - IoG (Intersection-over-GT) = area(P ∩ G) / area(G)
        This captures "GT is covered / contained by detection" even when detection is larger.

    Inputs:
      gt_objs, det_objs: output of load_objects_from_json()
    Returns:
      metrics dict with:
        - overall: {n_gt, n_det, mean_best_iog, mean_best_iou, coverage_rate@...}
        - (optional) per_severity: same structure for severity groups
        - (optional) per_type: same structure for anomaly type groups
    """
    from shapely.prepared import prep
    def best_scores_for_group(gt_list, det_list):
        if not gt_list:
            out = {"n_gt": 0, "n_det": len(det_list), "mean_best_iog": 0.0, "mean_best_iou": 0.0}
            for t in iog_thresholds:
                out[f"coverage_rate_iog>={t}"] = 0.0
            return out

        best_iog = []
        best_iou = []

        det_geoms = [d["geom"] for d in det_list]

        # (optional) prepare detections for faster intersects/contains
        det_prepared = [prep(g) for g in det_geoms]

        for g in gt_list:
            G = g["geom"]
            g_area = G.area
            if g_area <= 0:
                best_iog.append(0.0)
                best_iou.append(0.0)
                continue

            biog = 0.0
            biou = 0.0

            gb = G.bounds  # (minx, miny, maxx, maxy)

            for P, Pp in zip(det_geoms, det_prepared):
                # Manual bbox reject (fast)
                pb = P.bounds
                if (pb[2] < gb[0]) or (pb[0] > gb[2]) or (pb[3] < gb[1]) or (pb[1] > gb[3]):
                    continue

                if not Pp.intersects(G):
                    continue

                # inter = G.intersection(P).area
                try:
                    inter = G.intersection(P).area
                except GEOSException:
                    # last-resort: try cleaning again then intersect
                    G2 = _clean_geom(G)
                    P2 = _clean_geom(P)
                    if G2 is None or P2 is None:
                        continue
                    try:
                        inter = G2.intersection(P2).area
                    except GEOSException:
                        continue




                if inter <= 0:
                    continue

                iog = inter / g_area
                if iog > biog:
                    biog = iog

                union = G.union(P).area
                if union > 0:
                    iou = inter / union
                    if iou > biou:
                        biou = iou

            best_iog.append(float(biog))
            best_iou.append(float(biou))

        best_iog = np.asarray(best_iog, dtype=float)
        best_iou = np.asarray(best_iou, dtype=float)

        out = {
            "n_gt": int(len(gt_list)),
            "n_det": int(len(det_list)),
            "mean_best_iog": float(np.mean(best_iog)) if len(best_iog) else 0.0,
            "mean_best_iou": float(np.mean(best_iou)) if len(best_iou) else 0.0,
        }
        for t in iog_thresholds:
            out[f"coverage_rate_iog>={t}"] = float(np.mean(best_iog >= t)) if len(best_iog) else 0.0
        return out



    # Overall
    metrics = {
        "overall": best_scores_for_group(gt_objs, det_objs),
    }

    # Group by severity
    if by_severity:
        sev_keys = sorted({g.get("severity") for g in gt_objs if g.get("severity") is not None})
        per_sev = {}
        for sev in sev_keys:
            gt_g = [g for g in gt_objs if g.get("severity") == sev]
            # detections also have severity in your JSON, but may be unreliable; you can filter or not
            det_g = det_objs  # keep all dets by default (recommended)
            per_sev[sev] = best_scores_for_group(gt_g, det_g)
        metrics["per_severity"] = per_sev

    # Group by anomaly type
    if by_type:
        type_keys = sorted({g.get("road_anomaly_type") for g in gt_objs if g.get("road_anomaly_type") is not None})
        per_type = {}
        for t in type_keys:
            gt_g = [g for g in gt_objs if g.get("road_anomaly_type") == t]
            det_g = det_objs  # keep all dets by default
            per_type[t] = best_scores_for_group(gt_g, det_g)
        metrics["per_type"] = per_type

    return metrics


# Patch plt.show() to avoid "Agg is non-interactive" warnings from utilities.py
plt.show = lambda *a, **k: None

from utilities import compare_anomaly_results  # uses IoU matching and returns KPIs【turn6:3†utilities.py†L60-L73】


EXP_RE = re.compile(
    r"result_steps(?P<steps>\d+)_scale(?P<scale>[0-9.]+)_decay(?P<decay>[0-9.]+)\.json$"
)


def discover_runs(data_dir: str):
    """Return list of dicts: {exp_tag, steps, scale, decay, det_json_path}."""
    runs = []
    for path in glob.glob(os.path.join(data_dir, "result_*.json")):
        base = os.path.basename(path)
        m = EXP_RE.match(base)
        if not m:
            # Skip any non-matching result files
            continue
        steps = int(m.group("steps"))
        scale = float(m.group("scale"))
        decay = float(m.group("decay"))
        exp_tag = f"steps{steps}_scale{scale:g}_decay{decay:g}"
        runs.append({
            "exp_tag": exp_tag,
            "steps": steps,
            "scale": scale,
            "decay": decay,
            "det_json_path": path,
        })
    # Stable ordering for plots
    runs.sort(key=lambda r: (r["decay"], r["scale"], r["steps"]))
    return runs


def write_csv(rows, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write (no runs discovered?).")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_curves(rows, out_dir: str, metric: str):
    """
    For each decay:
      x = steps, lines = scale
    """
    os.makedirs(out_dir, exist_ok=True)

    # group rows by decay then by scale
    by_decay = defaultdict(list)
    for r in rows:
        by_decay[float(r["decay"])].append(r)

    for decay, items in sorted(by_decay.items()):
        # group by scale
        by_scale = defaultdict(list)
        for r in items:
            by_scale[float(r["scale"])].append(r)

        plt.figure(figsize=(7, 4))
        for scale, si in sorted(by_scale.items()):
            si_sorted = sorted(si, key=lambda x: int(x["steps"]))
            x = [int(s["steps"]) for s in si_sorted]
            y = [float(s.get(metric, "nan")) for s in si_sorted]
            plt.plot(x, y, marker="o", label=f"scale={scale:g}")

        plt.xlabel("SIM_STEPS (s)")
        plt.ylabel(metric)
        plt.ylim(0.6, 1.0)
        plt.title(f"{metric} vs SIM_STEPS (decay={decay:g})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"curve_{metric}_decay{decay:g}.png"), dpi=300)
        plt.close()


def plot_heatmap_final(rows, out_dir: str, metric: str):
    """
    Heatmap at max steps:
      y = decay, x = scale, value = metric (mean if duplicates)
    """
    os.makedirs(out_dir, exist_ok=True)

    steps_all = sorted({int(r["steps"]) for r in rows})
    if not steps_all:
        return
    max_steps = max(steps_all)

    final = [r for r in rows if int(r["steps"]) == max_steps]
    if not final:
        return

    decays = sorted({float(r["decay"]) for r in final})
    scales = sorted({float(r["scale"]) for r in final})

    mat = np.full((len(decays), len(scales)), np.nan, dtype=float)

    # if duplicates exist, take mean
    bucket = defaultdict(list)
    for r in final:
        bucket[(float(r["decay"]), float(r["scale"]))].append(float(r.get(metric, "nan")))

    for i, d in enumerate(decays):
        for j, s in enumerate(scales):
            vals = bucket.get((d, s), [])
            if vals:
                mat[i, j] = float(np.nanmean(vals))

    plt.figure(figsize=(6, 4))
    # plt.imshow(mat, origin="lower", aspect="auto")
    plt.xticks(range(len(scales)), [f"{s:g}" for s in scales])
    plt.yticks(range(len(decays)), [f"{d:g}" for d in decays])
    plt.xlabel("Flow scale")
    plt.ylabel("Decay rate")
    plt.title(f"{metric} at SIM_STEPS={max_steps}")
    im = plt.imshow(
        mat,
        cmap='viridis',
        vmin=0.0,    # <-- FIX MIN
        vmax=0.1     # <-- FIX MAX
    )
    plt.colorbar(im, label=metric)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_{metric}_steps{max_steps}.png"), dpi=300)
    plt.close()

def plot_heatmaps_for_steps(rows, out_dir: str, metric: str, steps_list):
    """
    Heatmap for each specified steps value:
      y = decay, x = scale, value = metric (mean if duplicates)
    Produces: heatmap_<metric>_steps<steps>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect available axes values globally so all heatmaps share the same layout
    all_decays = sorted({float(r["decay"]) for r in rows}, reverse=True)  # show high decay on top
    all_scales = sorted({float(r["scale"]) for r in rows})

    for steps in steps_list:
        subset = [r for r in rows if int(r["steps"]) == int(steps)]
        if not subset:
            print(f"[WARN] No rows for steps={steps}; skipping heatmap for {metric}")
            continue

        mat = np.full((len(all_decays), len(all_scales)), np.nan, dtype=float)
        bucket = defaultdict(list)

        for r in subset:
            bucket[(float(r["decay"]), float(r["scale"]))].append(float(r.get(metric, "nan")))

        for i, d in enumerate(all_decays):
            for j, s in enumerate(all_scales):
                vals = bucket.get((d, s), [])
                if vals:
                    mat[i, j] = float(np.nanmean(vals))

        plt.figure(figsize=(6, 4))
        # plt.imshow(mat, origin="upper", aspect="auto")  # origin upper because we reversed decays
        plt.xticks(range(len(all_scales)), [f"{s:g}" for s in all_scales])
        plt.yticks(range(len(all_decays)), [f"{d:g}" for d in all_decays])
        plt.xlabel("Flow scale")
        plt.ylabel("Decay rate")
        plt.title(f"{metric} at SIM_STEPS={steps}")
        im = plt.imshow(
            mat,
            cmap='viridis',
            vmin=0.0,    # <-- FIX MIN
            vmax=0.1     # <-- FIX MAX
        )
        plt.colorbar(im, label=metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_{metric}_steps{steps}.png"), dpi=300)
        plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="SCENARIO_NAME (brussels_rural/Graz_A2)")
    ap.add_argument("--data_root", default="data", help="Root folder for data (default: data)")
    ap.add_argument("--image_root", default="image", help="Root folder for images (default: image)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match (default: 0.5)")
    ap.add_argument("--skip_compare_plots", action="store_true",
                    help="Skip generating compare_*.png (faster). Metrics still computed.")
    args = ap.parse_args()

    scenario = args.scenario
    data_dir = os.path.join(args.data_root, scenario)
    img_dir = os.path.join(args.image_root, scenario)
    out_dir = os.path.join(img_dir, "convergence_summary")

    gt_json = os.path.join(data_dir, "damage_model.json")
    if not os.path.isfile(gt_json):
        raise FileNotFoundError(f"Ground truth not found: {gt_json}")
    
    runs = discover_runs(data_dir)
    if not runs:
        raise RuntimeError(
            f"No run results found in {data_dir}. "
            f"Expected files like result_steps3600_scale1.5_decay0.05.json"
        )


    gt_objs = load_objects_from_json(gt_json)
    rows = []
    for r in runs:
        exp_tag = r["exp_tag"]
        det_json = r["det_json_path"]

        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"compare_{exp_tag}.png")
        #############################################################



        # Recompute evaluation KPIs using IoU matching【turn6:3†utilities.py†L60-L73】
        res = compare_anomaly_results(
            gt_json_path=gt_json,
            det_json_path=det_json,
            save_path=save_path,
            iou_threshold=args.iou,
            net_file=None
        )

        overall = res.get("overall", {})

        # Coverage metrics (IoG)
        det_objs = load_objects_from_json(det_json)
        cov = compute_coverage_metrics(
            gt_objs, det_objs,
            iog_thresholds=(0.5, 0.8, 0.95),
            by_severity=False,
            by_type=False
        )
        overall_cov = cov["overall"]

        row = {
            "exp_tag": exp_tag,
            "steps": r["steps"],
            "scale": r["scale"],
            "decay": r["decay"],

            # IoU-threshold KPIs
            "tp": overall.get("true_positives", 0),
            "fp": overall.get("false_positives", 0),
            "fn": overall.get("false_negatives", 0),
            "precision": overall.get("precision", 0),
            "recall": overall.get("recall", 0),
            "f1": overall.get("f1_score", 0),
            "avg_iou": overall.get("avg_iou", 0),

            # Coverage KPIs
            "mean_best_iog": overall_cov["mean_best_iog"],
            "coverage_iog50": overall_cov["coverage_rate_iog>=0.5"],
            "coverage_iog80": overall_cov["coverage_rate_iog>=0.8"],
            "coverage_iog95": overall_cov["coverage_rate_iog>=0.95"],
            "mean_best_iou_gt": overall_cov["mean_best_iou"],
        }

        rows.append(row)


    # Save metrics table
    csv_path = os.path.join(data_dir, "convergence_metrics.csv")
    write_csv(rows, csv_path)
    print(f"[OK] Wrote metrics: {csv_path}")

    # Produce convergence plots
    os.makedirs(out_dir, exist_ok=True)

    # Primary: avg_iou and f1
    plot_curves(rows, out_dir, "avg_iou")
    plot_curves(rows, out_dir, "mean_best_iog")
    plot_curves(rows, out_dir, "coverage_iog80")
    plot_curves(rows, out_dir, "coverage_iog95")
    plot_curves(rows, out_dir, "precision")
    plot_curves(rows, out_dir, "recall")
    plot_curves(rows, out_dir, "f1_score")
    plot_curves(rows, out_dir, "avg_iou")


    # # Heatmaps at max steps
    plot_heatmap_final(rows, out_dir, "avg_iou")
    plot_heatmap_final(rows, out_dir, "coverage_iog80")
    # Heatmaps for selected steps (edit this list as you want)
    HEATMAP_STEPS = [300, 5400, 9000]

    plot_heatmaps_for_steps(rows, out_dir, "avg_iou", HEATMAP_STEPS)
    plot_heatmaps_for_steps(rows, out_dir, "coverage_iog80", HEATMAP_STEPS)


    print(f"[OK] Wrote plots to: {out_dir}")


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    import sys

    # If no CLI arguments are given (e.g., clicking Run in Cursor),
    # provide a default scenario here.
    if len(sys.argv) == 1:
        # ---- EDIT THIS ----
        DEFAULT_SCENARIO = "Graz_A2" #brussels_rural/Graz_A2
        sys.argv.extend(["--scenario", DEFAULT_SCENARIO])

    main()

