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
