# crowdsourced_road_damage_estimation



## Getting started

# Conda Environment

Go to your repository:

`cd crowdsourced_road_damage_estimation/env`

Create the environment with Conda:

`conda env create -n crowdsourced_road_damage_estimation -f environment.yml`

After successful installation, activate the environment with:

`conda activate crowdsourced_road_damage_estimation`

To use the environment later:

`conda activate crowdsourced_road_damage_estimation`

## TODOs
Here are TODOs for the project:
-  A model of hitting the road damage with a car according to the geometry of the road damage, pothole...
-  Extending the probability of detection, false alarms, and mapped from (low, medium, high) to different probabilities
-  Improving the detection of raod damage with a car
- currently it is detected per time steps, and there is a chance that the cars drives over the region of raod damage but no detection
here is the code for detection:
`def contains(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) falls within this damage's shape.
        """
        return self.shape.contains(Point(x, y))`

`            detection = sensor_model.detect_damage_position(step, x, y)
`

## Research questions
- the influence of road damage geometry on the detection of road damage and generation of the map (single lane road)
- the influence of severity of road damage on the detection of road damage and generation of the map (low, medium, high)
- the influence of the lane where the road damage is located on the detection of road damage and generation of the map (e.g. left lane, right lane, middle lane)
- How many and percentage of vehicles and how many of the data are needed to detect road damage with a certain probability?
