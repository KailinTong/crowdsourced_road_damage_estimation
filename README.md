# Crowdsourced Road Damage Estimation

Estimation of road damage (potholes) using Bayesian fusion of crowdsourced vehicle sensor data and InSAR satellite priors.

## 🚀 Getting Started

### 1. Environment Setup
Activate the conda environment:
```bash
conda activate crowdsourced_road_damage_estimation
```

## 🔬 Research Questions
- How does road damage **geometry** influence detection and map generation (e.g., single-lane vs. multi-lane)?
- How does the **prior probability** influence the convergence of the algorithm?
- How does the **simulation time** influence the final result?
- How does the **decay rate** influence the prediction result?
- Does the **lane location** (left, right, middle) of the damage impact the mapping accuracy?
- What is the minimum **vehicle density** (count and percentage) required for reliable detection?

## 🧪 Running Simulations (Manual Execution)

The project supports multiple scenarios. You can run them manually using the `--config` and `--mode` flags.

### Graz A2 Scenario
This scenario uses InSAR satellite priors mapped to scientific standards ([**ASTM D6433**](https://www.astm.org/d6433-23.html)). It uses millimeter-scale satellite displacement data to classify pothole severity (High >50mm, Medium 25-50mm, Low <25mm).
- **Simulate:** `python3 main.py --config config/Graz_A2_config.json --mode simulate`
- **Analyze:** `python3 main.py --config config/Graz_A2_config.json --mode analyze`

### Brussels Rural Scenario
- **Simulate:** `python3 main.py --config brussels_rural_config.json --mode simulate`
- **Analyze:** `python3 main.py --config brussels_rural_config.json --mode analyze`

*Add `--gui` to any simulation command to watch the vehicles in real-time.*

## 📈 Batch Experiments
To run automated simulations and analysis for both scenarios in sequence:
```bash
python3 run_experiments.py
```

## 🛰 Experimental Data (Graz A2 "Story")
The Graz A2 scenario features a specialized pipeline connecting satellite measurements to simulation priors:

- **Verified Mapping Logic ([ASTM D6433](https://www.astm.org/d6433-23.html)):**
  This scenario utilizes InSAR data from Graz (mm/year) to estimate pothole depth over a **5-year cumulative period**. The estimated cumulative displacement is mapped directly to the **ASTM D6433** scientific standard for pothole severity. To ensure data quality, we apply a **13mm minimum depth filter** (the ASTM threshold for 'Low' severity); anything shallower is considered surface noise and excluded:
    - **High Severity:** > 50mm estimated depth (Red)
    - **Medium Severity:** 25-50mm estimated depth (Orange)
    - **Low Severity:** 13-25mm estimated depth (Yellow)
    - *Note: < 13mm zones are filtered as non-potholes.*
- **Adjustable Priors:** Modify `UNIFORM_PRIOR` in `generate_graz_priors.py` (default: `0.6`) and run:
  ```bash
  python3 generate_graz_priors.py
  ```

## 📊 Evaluation & KPIs
You can evaluate the system's performance using standard ML and road engineering KPIs:
- **Run Evaluation:**
  ```bash
  python3 evaluate_detections.py --gt data/Graz_A2/road_anomaly_probabilities.json --pred data/Graz_A2/result_30.json
  ```
- **KPIs include:**
    - **TP / FP / FN:** Based on an **IoU threshold of 0.3** (common for irregular road anomalies).
    - **Precision, Recall, F1:** Standard ML detection metrics.
    - **Average IoU:** Measures spatial accuracy of the predicted bounds.
    - **Severity Accuracy:** Measures the accuracy of Low/Medium/High severity classification.


## 📊 Visualization
Visualize the prior maps and risk zones:
- **Clustered Map (Graz):** `python3 visualize_clustered_prior.py`
- **Risk Map (Graz):** `python3 visualize_risk_map.py`
- **Prior Map (Brussels):** `python3 visualize_prior_map.py`
- **Matches Plot:** Generates `image/<SCENARIO>/compare_<STEPS>.png` during evaluation.
