# crowdsourced_road_damage_estimation



## Getting started

### Conda Environment

Go to your repository:

`cd crowdsourced_road_damage_estimation/env`

Create the environment with Conda:

`conda env create -n crowdsourced_road_damage_estimation -f environment.yml`

After successful installation, activate the environment with:

`conda activate crowdsourced_road_damage_estimation`

To use the environment later:

`conda activate crowdsourced_road_damage_estimation`

### Start Simulation

 `python main.py -c brussels_rural_config.json -m simulate`

### Start Analyze

 `python main.py -c brussels_rural_config.json -m analyze`


## Research questions
- the influence of road damage geometry on the detection of road damage and generation of the map (single lane road)
- the influence of severity of road damage on the detection of road damage and generation of the map (low, medium, high)
- the influence of the lane where the road damage is located on the detection of road damage and generation of the map (e.g. left lane, right lane, middle lane)

- How many and percentage of vehicles and how many of the data are needed to detect road damage with a certain probability?

## Graz A2 Scenario (New)

### 1. Visualization
Visualize the risk zones overlaid on the road network:
`python visualize_risk_map.py`

Output: `image/Graz_A2_risk_visualization.png`

### 2. Batch Experiments
Run simulations for multiple scenarios (Graz_A2, brussels_rural) in sequence:
`python run_experiments.py`

This script will:
- Generate temporary configs with specified `SIM_STEPS`.
- Run simulation and analysis for each scenario.
- Save output images to `image/<scenario_name>/`.


### 3. Manual Execution (Graz A2)
**Simulate:**
`python main.py --config config/Graz_A2_config.json --mode simulate`

**Simulate with GUI:**
`python main.py --config config/Graz_A2_config.json --mode simulate --gui`

**Analyze:**
`python main.py --config config/Graz_A2_config.json --mode analyze`
*Note: Analysis loads risk priors from `data/Graz_A2/road_anomaly_probabilities.json`.*
