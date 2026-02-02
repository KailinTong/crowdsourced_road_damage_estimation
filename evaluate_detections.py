import json
import argparse
from shapely.wkt import loads as load_wkt
from shapely.geometry import shape, Polygon
import numpy as np
import os

# Literature Reference for Evaluation Criteria:
# Common road damage detection challenges (e.g., GRDDC) and literature on automated pavement distress 
# survey typically use F1-Score as the primary metric. 
# Spatial matching for irregular road anomalies like potholes often uses an IoU (Intersection over Union) 
# threshold lower than the standard 0.5 (often 0.3) due to the amorphous nature of ground truth and 
# the variability in sensing (crowdsourced GPS/accelerometer).
# Successful detection is defined as a predicted region that overlaps with ground truth above the threshold.
# In this implementation, we also evaluate severity accuracy for matched detections.

def evaluate(gt_path, pred_path, iou_threshold=0.3):
    """
    Evaluate predicted road anomalies against ground truth.
    
    Args:
        gt_path (str): Path to road_anomaly_probabilities.json (Ground Truth / Prior).
        pred_path (str): Path to result_*.json (Simulation Results).
        iou_threshold (float): IoU threshold for a True Positive match.
        
    Returns:
        dict: Evaluation metrics and KPIs.
    """
    if not os.path.exists(gt_path):
        return {"error": f"Ground truth file not found: {gt_path}"}
    if not os.path.exists(pred_path):
        return {"error": f"Prediction file not found: {pred_path}"}

    # Load GT
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)
    
    # Load Pred
    with open(pred_path, 'r') as f:
        pred_list = json.load(f)

    # Convert GT to a list of dicts with shapely polygons
    gt_polys = []
    for k, v in gt_dict.items():
        sev = v['severity']
        # Convert mild_road to a type if needed, but usually we only evaluate potholes
        if sev == 'mild_road':
            continue
        gt_type = f"pothole_{sev}"
        try:
            poly = load_wkt(v['polygon'])
            if not poly.is_valid:
                poly = poly.buffer(0)
            gt_polys.append({
                'id': k, 
                'type': gt_type, 
                'severity': sev, 
                'poly': poly,
                'area': poly.area
            })
        except Exception as e:
            print(f"Warning: Could not parse GT polygon for {k}: {e}")

    # Convert Pred to a list of dicts with shapely polygons
    pred_polys = []
    for i, p in enumerate(pred_list):
        try:
            poly = load_wkt(p['shape'])
            if not poly.is_valid:
                poly = poly.buffer(0)
            pred_polys.append({
                'id': p['id'], 
                'type': p['road_anomaly_type'], 
                'severity': p['severity'], 
                'poly': poly,
                'area': poly.area,
                'probability': p['probability']
            })
        except Exception as e:
            print(f"Warning: Could not parse Pred polygon for {p.get('id', i)}: {e}")

    # Matching logic: Greedy matching based on highest IoU
    matches = []
    used_gt = set()
    
    # Sort predictions by probability or area? Usually probability or just iterate
    # We'll use a matching matrix approach for clarity
    num_gt = len(gt_polys)
    num_pred = len(pred_polys)
    
    iou_matrix = np.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            p_poly = pred_polys[i]['poly']
            g_poly = gt_polys[j]['poly']
            
            if p_poly.intersects(g_poly):
                intersection = p_poly.intersection(g_poly).area
                union = p_poly.union(g_poly).area
                iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Greedy assignment
    # We want to match pairs with the highest IoU first
    indices = np.argsort(iou_matrix.ravel())[::-1]
    matched_preds = set()
    matched_gts = set()
    
    for idx in indices:
        p_idx, g_idx = divmod(idx, num_gt)
        if iou_matrix[p_idx, g_idx] < iou_threshold:
            break
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            
            p = pred_polys[p_idx]
            g = gt_polys[g_idx]
            
            matches.append({
                'pred_id': p['id'],
                'gt_id': g['id'],
                'iou': iou_matrix[p_idx, g_idx],
                'pred_severity': p['severity'],
                'gt_severity': g['severity'],
                'severity_match': p['severity'] == g['severity']
            })

    # KPIs Calculation
    TP = len(matches)
    FP = num_pred - TP
    FN = num_gt - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_iou = np.mean([m['iou'] for m in matches]) if matches else 0
    severity_accuracy = np.mean([m['severity_match'] for m in matches]) if matches else 0

    # Successful detection defined as matching GT with IoU >= threshold
    successful_detections_count = TP

    results = {
        'scenario_info': {
            'gt_file': gt_path,
            'pred_file': pred_path,
            'iou_threshold': iou_threshold
        },
        'counts': {
            'ground_truth_total': num_gt,
            'predictions_total': num_pred,
            'true_positives': TP,
            'false_positives': FP,
            'false_negatives': FN,
        },
        'metrics': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'average_iou': round(avg_iou, 4),
            'severity_accuracy': round(severity_accuracy, 4)
        },
        'successful_detection_criteria': f"Matching Ground Truth with IoU >= {iou_threshold}",
        'successful_detections': successful_detections_count
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate road damage detection results using common ML KPIs.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth JSON (road_anomaly_probabilities.json).")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction JSON (result_*.json).")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching (default: 0.3).")
    parser.add_argument("--output", type=str, help="Optional path to save evaluation results as JSON.")
    
    args = parser.parse_args()
    
    res = evaluate(args.gt, args.pred, args.iou)
    
    if "error" in res:
        print(f"Error: {res['error']}")
    else:
        print("\n=== Road Damage Estimation Evaluation ===")
        print(f"GT File: {args.gt}")
        print(f"Pred File: {args.pred}")
        print("-" * 40)
        print(f"Ground Truth Total: {res['counts']['ground_truth_total']}")
        print(f"Predictions Total:  {res['counts']['predictions_total']}")
        print(f"True Positives:     {res['counts']['true_positives']}")
        print(f"False Positives:    {res['counts']['false_positives']}")
        print(f"False Negatives:    {res['counts']['false_negatives']}")
        print("-" * 40)
        print(f"Precision:          {res['metrics']['precision']}")
        print(f"Recall:             {res['metrics']['recall']}")
        print(f"F1-Score:           {res['metrics']['f1_score']}")
        print(f"Average IoU:        {res['metrics']['average_iou']}")
        print(f"Severity Accuracy:  {res['metrics']['severity_accuracy']}")
        print("-" * 40)
        print(f"Successful Detections: {res['successful_detections']}")
        print("==========================================\n")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(res, f, indent=4)
            print(f"Results saved to {args.output}")
