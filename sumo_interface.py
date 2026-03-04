import traci
import traci.constants as tc
class SumoInterface:
    def __init__(self, sumo_cmd):
        """
        Initialize SUMO via TraCI. sumo_cmd: list of command-line args.
        """
        traci.start(sumo_cmd)

    def step(self):
        """Advance one simulation step."""
        traci.simulationStep()

    def get_vehicle_positions(self):
        """
        Return dict of {veh_id: (x, y)} in world coords.
        """
        positions = {}
        for vid in traci.vehicle.getIDList():
            x = traci.vehicle.getPosition(vid)[0]
            y = traci.vehicle.getPosition(vid)[1]
            positions[vid] = (x, y)
        return positions

    def get_vehicle_type(self, veh_id):
        """
        Return the vehicle type of the given vehicle ID.
        """
        return traci.vehicle.getTypeID(veh_id)

    def close(self):
        traci.close()
